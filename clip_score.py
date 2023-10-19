import clip
import os
from PIL import Image
import torch
import pandas as pd

style =    'melting golden 3d rendering style'
img_path = 'outcomes/generation/melting_golden_3d_rendering/WithoutStyleEdit/new1'
gt_path =  'dataset/melting_golden/r1/image_03_07.jpg'
target_path = 'outcomes/clip_score'

class ClipScore():
    def __init__(self, img_path, gt_path, style ,target_path="",model_name='ViT-B/32', device='cuda'):
        self.model_name = model_name
        self.device = device
        self.img_path = img_path
        self.style = style
        #score clip scores
        index = img_path.index('/')
        img_path_ = img_path[index+1:]
        index = img_path_.index('/')
        self.target_name = img_path_[index+1:].replace('/', '_')
        self.csv_name = self.target_name+'.csv'
        self.csv_k_name = self.target_name+'_topk.csv'
        self.txt_name = self.target_name+'.txt'
        if len(target_path) == 0:
            self.target_path = os.path.join(img_path, '..')
            self.target_path = os.path.join(self.target_path, '..')
            self.target_path = os.path.join(self.target_path, 'clip_score')
            if not os.path.exists(self.target_path):
                os.mkdir(self.target_path)
            self.target_path = os.path.join(self.target_path, self.target_name)
            if not os.path.exists(self.target_path):
                os.mkdir(self.target_path)
        else:
            self.target_path = os.path.join(target_path, self.target_name)
            if not os.path.exists(self.target_path):
                os.mkdir(self.target_path)
        
        #load data
        self._load_data(gt_path, img_path)
        #load model
        self._load_model()
        
    
    def _load_model(self):
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        self.tokenizer = clip.tokenize
        self.logit_scale = self.model.logit_scale.exp()

    def _load_data(self, gt_path, img_path):
        self.gt = Image.open(gt_path)
        self.contents = os.listdir(img_path)
        
        self.imgs = []
        self.img_names = []
        self.prompt_contents = []
        self.prompts = []
        for content in self.contents:
            path1 = os.path.join(img_path, content)
            prompt_content = content.replace('_', ' ')
            prompt = prompt_content + ' in ' + self.style
            
            img_files = os.listdir(path1)
            for img_file in img_files:
                img = Image.open(os.path.join(path1, img_file))
                self.imgs.append(img)
                
                self.prompt_contents.append(prompt_content)
                self.prompts.append(prompt)
                self.img_names.append(prompt+'_'+img_file)
            
        self.len = len(self.img_names)

    def _text_embed(self, text):
        text_tensor = self.tokenizer(text).to(self.device)
        text_feature = self.model.encode_text(text_tensor)
        return text_feature
    
    def _img_embed(self, im):
        im = self.preprocess(im)
        im = im.unsqueeze(0).to(self.device)
        im_features = self.model.encode_image(im)
        return im_features
        
    def _get_score(self, a, b):
        a = a/a.norm(dim=1, keepdim=True).to(torch.float32)
        b = b/b.norm(dim=1, keepdim=True).to(torch.float32)
        score = self.logit_scale * (a * b).sum()
        return score

    def _text_score(self):
        im = self.imgs[self.index]
        score_prompt = self._get_score(self._img_embed(im), self._text_embed(self.prompts[self.index]))
        score_content = self._get_score(self._img_embed(im), self._text_embed(self.prompt_contents[self.index]))
        score_style = self._get_score(self._img_embed(im), self._text_embed(self.style))
        
        return score_prompt, score_content, score_style
    
    def _img_score(self):
        im = self.imgs[self.index]
        return self._get_score(self._img_embed(im), self._img_embed(self.gt))
    
    def run(self, print_=False):
        scores = []
        for i in range(self.len):
            self.index = i
            img_score = self._img_score()
            text_prompt_score, text_content_score, text_style_score = self._text_score()
            scores.append([self.img_names[i], img_score.item(), text_prompt_score.item(), text_content_score.item(), text_style_score.item()])
            """
            if print_:
                print('index: (%d/%d), img: %s, style score: %.3f, text_score: %.3f'%(i, self.len, self.img_names[i], style_score, text_score))
            """
        self.columns = ['name', 'img score', 'text prompt score', 'text content score', 'text style score']
        scores = pd.DataFrame(scores, columns=self.columns)
        scores.to_csv(os.path.join(self.target_path, self.csv_name))
        self.scores = scores
    
    def get_top_k(self, print_=True, k=5, mode=0):
        target = self.scores[self.columns[mode+1]].values.tolist()
        
        sorted_target = sorted(target)
        topk_items = sorted_target[-k:]
        topk_indexes = [target.index(m) for m in topk_items]
        clip_list = [self.img_names[m] for m in topk_indexes]
        if print_:
            print(clip_list)
        #Print clip-score list(csv)
        top_k = [[a[:-4], b] for (a, b) in zip(clip_list, topk_items)]
        top_k = pd.DataFrame(top_k, columns=['names', 'scores'])
        top_k.to_csv(os.path.join(self.target_path, self.csv_k_name),index=False)
        #Print clip list
        f = open(os.path.join(self.target_path, self.txt_name), 'w')
        for i, clip in enumerate(clip_list):
            if i == 0:
                f.write("[")
            f.write("'"+clip+"'")
            if i != k-1:
                f.write(',')
            else:
                f.write(']')
        
if __name__ == '__main__':
    score = ClipScore(img_path, gt_path, style, target_path)
    score.run(True)
    score.get_top_k(k=5, mode=0)