import torch
import torch.nn as nn
from torch.nn import Parameter
import os
import numpy as np

class PromptRefiner():
    def __init__(self, PRF_mode=None, k_content = 0, k_style=5, accelerator=None):
        super().__init__()
        self.PRF_mode = PRF_mode
        self.mode = 'train'
        
        self.placeholder_tokens = ['*', '#','&','?','@']
        self.k_content = k_content
        self.k_style = k_style
        self.accelerator = accelerator
        
        self.placeholder_tokens_content=None
        self.placeholder_tokens_style=None
    
    def _get_token_embds(self, index):
        text_encoder = self.encoders[index]
        tokenizer = self.tokenizers[index]
        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data
        return token_embeds
    
    def _set_encoder_grad(self, index):
        text_encoder = self.encoders[index]
        text_encoder.text_model.encoder.requires_grad_(False)
        text_encoder.text_model.final_layer_norm.requires_grad_(False)
        text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    
    def Initial(self, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2):
        self.tokenizers = [tokenizer_1, tokenizer_2]
        self.encoders = [text_encoder_1, text_encoder_2]
    
    def set_mode(self, mode):
        # mode in ['train', 'infer', 'lock']
        #lock: train lora without PRF
        self.mode = mode
        return self.mode
    
    def get_placeholders(self, get_all=False):
        return self.placeholder_tokens_content, self.placeholder_tokens_style
    
    def get_tokenizers(self):
        return self.tokenizers
    
    def get_encoders(self):
        return self.encoders
    
    def _get_min(self, index):
        ret = min(self.placeholder_tokens_ids_style[index])
        if self.k_content >0:
            x1 = min(self.placeholder_tokens_ids_content[index])
            ret = ret if ret < x1 else x1
        return ret

    def _get_max(self, index):
        ret = max(self.placeholder_tokens_ids_style[index])
        if self.k_content >0:
            x1 = max(self.placeholder_tokens_ids_content[index])
            ret = ret if ret > x1 else x1
        return ret    
    
    def _step(self, index):
        #print('index_no_updates',self._get_min(index),self._get_max(index) + 1)
        tokenizer = self.tokenizers[index]
        index_no_updates = torch.ones((len(tokenizer), ), dtype=torch.bool)
        index_no_updates[self._get_min(index):self._get_max(index) + 1] = False

        with torch.no_grad():
            self.accelerator.unwrap_model(self.encoders[index]).get_input_embeddings().weight[
                index_no_updates
            ] = self.orig_embeds_params[index][index_no_updates]
    
    def step(self):
        self._step(0)
        self._step(1)
    
    def save_org_embeds_params(self):
        self.orig_embeds_params = [self.accelerator.unwrap_model(self.encoders[0]).get_input_embeddings().weight.data.clone(),
                                   self.accelerator.unwrap_model(self.encoders[1]).get_input_embeddings().weight.data.clone()]            
    
    def get_saved_path(self, output_dir, check_point):
        dirs_list = output_dir.split('/')
        PRF_dir = os.path.join(dirs_list[0], dirs_list[1]+'_encoders')
        if (not os.path.exists(PRF_dir)):
            os.mkdir(PRF_dir)
        PRF_path = os.path.join(PRF_dir, check_point)
        if (not os.path.exists(PRF_path)):
            os.mkdir(PRF_path)
        enc_path_1 = os.path.join(PRF_path, 'enc_1.pth')
        enc_path_2 = os.path.join(PRF_path, 'enc_2.pth')

        return [enc_path_1, enc_path_2], PRF_path
    
    def prepare(self):
        #set placeholders
        if self.k_content >0:
            placeholder_tokens_content = [] 
        placeholder_tokens_style = []
        if self.k_content >= 1:
            for i in range(0, self.k_content):
                placeholder_tokens_content.append(f"{self.placeholder_tokens[0]}_{i}")
        for i in range(0, self.k_style):
            placeholder_tokens_style.append(f"{self.placeholder_tokens[1]}_{i}")
        
        #add placeholders to tokenizer        
        if self.k_content > 0:
            self.placeholder_tokens_content = placeholder_tokens_content
            num_added_tokens_1 = self.tokenizers[0].add_tokens(placeholder_tokens_content)
            num_added_tokens_2 = self.tokenizers[1].add_tokens(placeholder_tokens_content)
            #assert num_added_tokens_1 == self.k_content, "content, num_added_tokens_1 ERROR!"
            #assert num_added_tokens_2 == self.k_content, "content, num_added_tokens_2 ERROR!"
        self.placeholder_tokens_style = placeholder_tokens_style
        num_added_tokens_1 = self.tokenizers[0].add_tokens(placeholder_tokens_style)
        num_added_tokens_2 = self.tokenizers[1].add_tokens(placeholder_tokens_style)
        #assert num_added_tokens_1 == self.k_style, "style, num_added_tokens_1 ERROR!"
        #assert num_added_tokens_2 == self.k_style, "style, num_added_tokens_2 ERROR!"
        
        if self.k_content >0:
            token_ids_content_1 = self.tokenizers[0].encode('content', add_special_tokens=False)
            token_ids_content_2 = self.tokenizers[1].encode('content', add_special_tokens=False)
        token_ids_style_1 = self.tokenizers[0].encode('style', add_special_tokens=False)
        token_ids_style_2 = self.tokenizers[1].encode('style', add_special_tokens=False)
        
        if self.k_content >0:
            ini_token_ids_content_1 = token_ids_content_1[0]
            ini_token_ids_content_2 = token_ids_content_2[0]
        ini_token_ids_style_1 = token_ids_style_1[0]
        ini_token_ids_style_2 = token_ids_style_2[0]
        
        if self.k_content >0:
            placeholder_token_ids_content_1 = self.tokenizers[0].convert_tokens_to_ids(placeholder_tokens_content)
            placeholder_token_ids_content_2 = self.tokenizers[1].convert_tokens_to_ids(placeholder_tokens_content)
            self.placeholder_token_ids_content = [placeholder_token_ids_content_1, placeholder_token_ids_content_2]
        placeholder_token_ids_style_1 = self.tokenizers[0].convert_tokens_to_ids(placeholder_tokens_style)
        #print('placeholder_tokens_style', placeholder_tokens_style)
        #print('placeholder_token_ids_style_1',placeholder_token_ids_style_1)
        placeholder_token_ids_style_2 = self.tokenizers[1].convert_tokens_to_ids(placeholder_tokens_style)
        self.placeholder_tokens_ids_style = [placeholder_token_ids_style_1, placeholder_token_ids_style_2]
        
        #set token embeds
        token_embeds_1 = self._get_token_embds(0)
        token_embeds_2 = self._get_token_embds(1)
        
        with torch.no_grad():
            for token_id in placeholder_token_ids_style_1:
                token_embeds_1[token_id] = token_embeds_1[ini_token_ids_style_1].clone()
            for token_id in placeholder_token_ids_style_2:
                token_embeds_2[token_id] = token_embeds_2[ini_token_ids_style_2].clone()
            if self.k_content > 0:
                for token_id in placeholder_token_ids_content_1:
                    token_embeds_1[token_id] = token_embeds_1[ini_token_ids_content_1].clone()
                for token_id in placeholder_token_ids_content_2:
                    token_embeds_2[token_id] = token_embeds_2[ini_token_ids_content_2].clone()

        self.token_embeds = [token_embeds_1, token_embeds_2]
        
        #set grad
        self._set_encoder_grad(0)
        self._set_encoder_grad(1)
        
        return self.token_embeds
    
    def pre_prompt(self, prompt):
        prompt = prompt.strip()
        prompt_list = prompt.split(' ')
        l = len(prompt_list)
        index_in = prompt_list.index('in')
        content_list = prompt_list[:index_in]
        style_list = prompt_list[index_in+1:-1]
        l_content = len(content_list)
        l_style = len(style_list)
        
        if self.PRF_mode == 1:
            #train/infer: A [C] in [S] [S*] style; len(S*) = self.k_style
            #lock [S*] = 'style' * self.k_style
            prompt_new = ""
            for i, c in enumerate(content_list):
                prompt_new += c
                prompt_new += ' '
            prompt_new += 'in '
            for i, s in enumerate(style_list):
                prompt_new += s 
                prompt_new += ' '
            if self.mode != 'lock':
                for i, s in enumerate(self.placeholder_tokens_style):
                    prompt_new += s
                    prompt_new += ' '
            else:
                for i in range(self.k_style):
                    prompt_new += 'style '
            prompt_new += 'style'
                
        return prompt_new

