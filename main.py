import copy
import imageio.v2 as imageio
import json
import numpy as np
import os
import random
import requests
import shutil
import tqdm
from PIL import Image
from itertools import combinations

#base_uri = "https://bafybeigtouebpxhlaps7f3mlvo7lwortfc7umg7dp2q6uut3muiwarslee.ipfs.nftstorage.link/%s.json"
total_number = (10000,1)
#name = "OkayBears"
base_uri = "https://moonrunners.herokuapp.com/api/%s.json"
name = "moonrunners"
mode = "eth"

class MetaCatcher:

    def __init__(self, base_uri, total_number, name, mode):
        self.base_uri = base_uri
        self.total_number = total_number[0]
        self.start_number = total_number[1]
        self.name = name
        self.mode = mode
        self.trait = {}

    def get_uri(self, uid):
        meta = requests.get(self.base_uri%uid).json()
        return meta

    def get_img(self, meta):
        if self.mode == "metaplex":
            ret = requests.get(meta['properties']['files'][0]['uri']).content
        elif self.mode == "eth":
            ret = requests.get(meta['image']).content
        return ret

    def get_trait(self, meta):
        return dict((i['trait_type'],i['value']) for i in meta['attributes'])

    def output_uri(self, uid, meta):
        with open('%s/meta/%s.json'%(self.name, uid), 'w') as f:
            f.write(json.dumps(meta, sort_keys=True, indent=4))

    def output_image(self, uid, content):
        with open('%s/img/%s.png'%(self.name, uid),'wb') as f:
            f.write(content)

    def download(self):
        if not os.path.exists(self.name):
            os.mkdir(self.name)
        if not os.path.exists(self.name+"/meta"):
            os.mkdir(self.name+"/meta")
        if not os.path.exists(name+"/img"):
            os.mkdir(self.name+"/img")
        for i in tqdm.tqdm(range(self.start_number,self.start_number+self.total_number)):
            if not (os.path.exists("%s/meta/%s.json"%(self.name, i)) and os.path.exists("%s/img/%s.png"%(self.name, i))):
                meta = self.get_uri(i)
                im = self.get_img(meta)
                self.output_uri(i, meta)
                self.output_image(i, im)
                self.trait[i] = self.get_trait(meta)
            else:
                with open("%s/meta/%s.json"%(self.name, i), 'r') as f:
                    meta = json.loads(f.read())
                self.trait[i] = self.get_trait(meta)
        with open('%s/trait.json'%self.name, 'w') as f:
            f.write(json.dumps(self.trait,sort_keys=True, indent=4))
        self.describe_all_trait()

    def load_trait(self):
        with open('%s/trait.json'%self.name, 'r') as f:
            self.trait = json.loads(f.read())

    def describe_trait(self, trait_name):
        trait_list = []
        if not self.trait:
            self.load_trait()
        for i in self.trait.keys():
            if trait_name in self.trait[i].keys():
                trait_list.append(self.trait[i][trait_name])
        trait_count = {}
        for i in set(trait_list):
            trait_count[i] = trait_list.count(i)
        return trait_count

    def describe_all_trait(self):
        trait_list = []
        if not self.trait:
            self.load_trait()
        for i in self.trait.keys():
            trait_list.extend(self.trait[i].keys())
        all_trait = {}
        for i in set(trait_list):
            all_trait[i] = self.describe_trait(i)
        with open('%s/all_trait.json'%self.name, 'w') as f:
            f.write(json.dumps(all_trait, sort_keys=True, indent=4))
        return all_trait

    def get_trait_set(self, trait_list, mute_trait=[], limit=200, output=False, clip=None):
        trait_set = []
        if not self.trait:
            self.load_trait()
        
        for i in self.trait.keys():
            is_same = True
            for t in mute_trait:
                if t in self.trait[i].keys():
                    is_same = False
            for t in trait_list.keys():
                if t not in self.trait[i].keys() or self.trait[i][t] != trait_list[t]:
                    is_same = False
            if is_same:
                trait_set.append(i)
        random.shuffle(trait_set)
        trait_set = trait_set[:limit]
        if output:
            if os.path.exists("train"):
                shutil.rmtree("train")
            os.mkdir("train")
            os.mkdir("train/output")
            for i in trait_set:
                if clip:
                    img = Image.open("%s/img/%s.png"%(self.name,i)).convert('RGB')
                    img = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3).astype(np.uint8)
                    if clip[0]:
                        img[:clip[0][0],:] = (255,255,255)
                        img[clip[0][1]:,:] = (255,255,255)
                    imageio.imwrite("train/output/%s.png"%i, img)
                else:
                    shutil.copy("%s/img/%s.png"%(self.name,i),"train/output/%s.png"%i)
        return trait_set

    def get_backgroundRGB(self):
        tc = self.describe_trait("Background")
        if not self.trait:
            self.load_trait()
        color_map = {}
        for i in tc.keys():
            for j in self.trait.keys():
                if self.trait[j]["Background"] == i:
                    im = imageio.imread("%s/img/%s.png"%(self.name,j))
                    color_map[i] = list(im[2,2,:])
                    break
        return color_map

    def train_trait(self, trait_name, control_trait=None, mute_trait=[], is_coverd=False,clip=None):
        tl = self.describe_all_trait()[trait_name]
        im = ImageMaster((300,300,3))
        if control_trait:
            cl = self.describe_all_trait()[control_trait]
            for c in cl:
                print(c)
                for i in tqdm.tqdm(tl):
                    if is_coverd:
                        tgt = mc.train_covered_trait({trait_name: i,
                                        control_trait: c}, mute_trait, im)
                        if not os.path.exists("component"):
                            os.mkdir("component")
                        if not os.path.exists("component/%s"%trait_name):
                            os.mkdir("component/%s"%trait_name)
                        if not os.path.exists("component/%s/%s"%(trait_name,c)):
                            os.mkdir("component/%s/%s"%(trait_name,c))
                        imageio.imwrite("component/%s/%s/%s.png"%(trait_name,c,i),tgt)
                    else:
                        mc.get_trait_set({trait_name: i,
                                        control_trait: c},
                                        mute_trait=mute_trait,
                                        output=True,
                                        clip=clip)
                        count = im.train()
                        if not os.path.exists("component"):
                            os.mkdir("component")
                        if not os.path.exists("component/%s"%trait_name):
                            os.mkdir("component/%s"%trait_name)
                        if not os.path.exists("component/%s/%s"%(trait_name,c)):
                            os.mkdir("component/%s/%s"%(trait_name,c))
                        if len(os.listdir("train/%s"%count)) > 0:
                            shutil.copy("train/%s/0.png"%count,"component/%s/%s/%s.png"%(trait_name,c,i))
        else:
            for i in tqdm.tqdm(tl):
                mc.get_trait_set({trait_name: i}, mute_trait=mute_trait, output=True)
                count = im.train()
                if not os.path.exists("component"):
                    os.mkdir("component")
                if not os.path.exists("component/%s"%trait_name):
                    os.mkdir("component/%s"%trait_name)
                shutil.copy("train/%s/0.png"%count,"component/%s/%s.png"%(trait_name,i))
        return

    def train_covered_trait(self, trait, coverd_list, im):
        iml = []
        for i in coverd_list:
            mc.get_trait_set(trait, mute_trait=[i], output=True)
            if len(os.listdir("train/output")) >= 2:
                count = im.enhanced_train()
                if len(os.listdir("train/%s"%count)) > 0:
                    iml.append(imageio.imread("train/%s/0.png"%count))
        tgt = iml[0]
        for i in iml:
            tgt = im.get_append(tgt, i)[0]
        return tgt
              
class ImageMaster:

    def __init__(self, shape):
        self.shape = shape
        return

    def get_imgdiff(self, i1, i2):
        diff = abs(i1-i2)
        diff = diff[:,:,0]+diff[:,:,1]+diff[:,:,2]
        diff = np.array([diff,diff,diff])
        diff = np.swapaxes(diff,0,1)
        diff = np.swapaxes(diff,1,2)
        return diff

    def fix_size(self, img):
        if img.shape != self.shape:
            img = np.vstack((img,img[-2:,:,:]))[:-1,:,:]
        return img

    def get_common(self, i1, i2):
        diff = self.get_imgdiff(i1,i2)
        i1 = copy.copy(i1)
        i2 = copy.copy(i2)
        i2[diff>0]=255
        return i2

    def get_append(self, i1, i2):
        i1t = copy.copy(i1)
        i2t = copy.copy(i2)
        i255 = np.ones(self.shape)*255
        i1t[i1==i255] = i2[i1==i255]
        i2t[i2==i255] = i1[i2==i255]
        return i1t, i2t

    def rand_name(self, fpath):
        count = 0
        flist = os.listdir(fpath)
        for i in flist:
            os.rename("%s/%s"%(fpath,i), "%s/%s_c.png"%(fpath,i))
        flist = os.listdir(fpath)
        random.shuffle(flist)
        for i in flist:
            os.rename("%s/%s"%(fpath,i), "%s/%s.png"%(fpath,count))
            count += 1

    def train1(self, src, tgt):
        ilist = os.listdir("train/%s"%src)
        if os.path.exists('train/%s'%tgt):
            shutil.rmtree('train/%s'%tgt)
        os.mkdir('train/%s'%tgt)
        icache = []
        for i in ilist:
            icache.append(self.fix_size(imageio.imread('train/%s/%s'%(src,i))))
            if len(icache)==2:
                imageio.imwrite("train/%s/s%s"%(tgt,i),self.get_common(*icache))
                icache = []
        self.rand_name("train/%s"%tgt)

    def train2(self, src,tgt):
        ilist = os.listdir("train/%s"%src)
        if os.path.exists('train/%s'%tgt):
            shutil.rmtree("train/%s"%tgt)
        os.mkdir("train/%s"%tgt)
        icache = []
        for i in ilist:
            icache.append(self.fix_size(imageio.imread('train/%s/%s'%(src,i))))
            if len(icache)==2:
                i1,i2 = self.get_append(*icache)
                imageio.imwrite("train/%s/a_%s"%(tgt,i),i1)
                imageio.imwrite("train/%s/b_%s"%(tgt,i),i2)
                icache = []
        self.rand_name("train/%s"%tgt)

    def generate_background(self, RGB):
        if not os.path.exists("component"):
            os.mkdir("component")
        if not os.path.exists("component/Background"):
            os.mkdir("component/Background")
        for i in RGB.keys():
            im = np.array([[RGB[i]]*self.shape[0]]*self.shape[1]).astype(np.uint8)
            imageio.imwrite("component/Background/%s.png"%i,im)
        return

    def turn_transparent(self, src):
        img = Image.open(src)
        img = img.convert("RGBA")
        datas = img.getdata()

        newData = []
        for item in datas:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)

        img.putdata(newData)
        img.save(src, "PNG")

    def turn_rgb(self, src):
        for i in os.listdir(src):
            fpath = src+"/"+i
            Image.open(fpath).convert('RGB').save(fpath)

    def train(self):
        self.turn_rgb("train/output")
        self.rand_name("train/output")
        count = 0
        self.train1("output", count)
        while len(os.listdir("train/%s"%count)) > 1:
            self.train2(count, count+1)
            count += 1
            self.train1(count, count+1)
            count += 1
        return count

    def enhanced_train(self):
        os.mkdir("train/enhanced")
        for i in range(16):
            count = self.train()
            shutil.copy("train/%s/0.png"%count,"train/enhanced/%s.png"%i)
        self.rand_name("train/enhanced")
        count = 0
        self.train1("enhanced", count)
        while len(os.listdir("train/%s"%count)) > 1:
            self.train2(count, count+1)
            count += 1
            self.train2(count, count+1)
            count += 1
            self.train1(count, count+1)
            count += 1
        return count

    def all_component_transparent(self):
        for i in os.listdir("component"):
            cl = os.listdir("component/%s"%i)
            if cl[0][-3:] == "png":
                for c in cl:
                    self.turn_transparent("component/%s/%s"%(i,c))
            else:
                for c in cl:
                    for t in os.listdir("component/%s/%s"%(i,c)):
                        self.turn_transparent("component/%s/%s/%s"%(i,c,t))

mc = MetaCatcher(base_uri, total_number, name, mode)
im = ImageMaster((300,300,3))
mc.train_trait("Legendary")
#im.all_component_transparent()