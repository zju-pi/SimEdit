import torch
from torchvision.transforms import Normalize
from transformers import CLIPProcessor, CLIPModel

class CLIP(torch.nn.Module):
    
    OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
    CLIP_SIZE = 224

    def __init__(self,
                 pretrain_model='openai/clip-vit-large-patch14'):
        super().__init__()
        model = CLIPModel.from_pretrained(pretrain_model)
        processor = CLIPProcessor.from_pretrained(pretrain_model)
        self.model = model
        self.tokenizer = processor.tokenizer
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((self.CLIP_SIZE, self.CLIP_SIZE))
        self.normalize = Normalize(mean=self.OPENAI_CLIP_MEAN,
                                   std=self.OPENAI_CLIP_STD)

    def forward(self, image1, text1):
        text = self.tokenizer([text1],
                              return_tensors="pt",
                              padding=True).to(image1.device)

        # resize image to 224x224
        image1 = self.avg_pool(image1)

        # normalize
        image1 = self.normalize(image1)
        
        image = image1.unsqueeze(0)
        text['pixel_values'] = image

        out = self.model(**text)

        image1_feat = out.image_embeds
        text1_feat = out.text_embeds

        similarity = torch.nn.CosineSimilarity()(image1_feat, text1_feat)
        return similarity
    
class DCLIPLoss(torch.nn.Module):
    
    OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
    CLIP_SIZE = 224

    def __init__(self,
                 pretrain_model='openai/clip-vit-base-patch32'):
        super().__init__()
        model = CLIPModel.from_pretrained(pretrain_model)
        processor = CLIPProcessor.from_pretrained(pretrain_model)
        self.model = model
        self.tokenizer = processor.tokenizer
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((self.CLIP_SIZE, self.CLIP_SIZE))
        self.normalize = Normalize(mean=self.OPENAI_CLIP_MEAN,
                                   std=self.OPENAI_CLIP_STD)

    def forward(self, image1, image2, text1, text2):
        text = self.tokenizer([text1, text2],
                              return_tensors="pt",
                              padding=True).to(image1.device)

        # resize image to 224x224
        image1 = self.avg_pool(image1)
        image2 = self.avg_pool(image2)

        # normalize
        image1 = self.normalize(image1)
        image2 = self.normalize(image2)
        
        image = torch.cat([image1.unsqueeze(0),
                           image2.unsqueeze(0)])
        text['pixel_values'] = image

        print(image.shape)

        print(image1.dtype)
        print(image2.dtype)

        out = self.model(**text)

        image1_feat = out.image_embeds[0:1]
        image2_feat = out.image_embeds[1:2]
        text1_feat = out.text_embeds[0:1]
        text2_feat = out.text_embeds[1:2]

        d_image_feat = image1_feat - image2_feat
        d_text_feat = text1_feat - text2_feat

        similarity = torch.nn.CosineSimilarity()(d_image_feat, d_text_feat)
        return 1 - similarity