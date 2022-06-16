import json
import os
import requests
import tqdm

base_uri = "https://bafybeigtouebpxhlaps7f3mlvo7lwortfc7umg7dp2q6uut3muiwarslee.ipfs.nftstorage.link/%s.json"
total_numbers = 10000
name = "OkayBears"

def get_uri(base_uri, uid):
    meta = requests.get(base_uri%uid).json()
    return meta

def get_img(meta):
    return requests.get(meta['properties']['files'][0]['uri']).content

def output_uri(name, uid, meta):
    with open('%s/meta/%s.json'%(name, uid), 'w') as f:
        f.write(json.dumps(meta))

def output_image(name, uid, content):
    with open('%s/img/%s.png'%(name, uid),'wb') as f:
        f.write(content)

def download(name, base_uri):
    if not os.path.exists(name):
        os.mkdir(name)
    if not os.path.exists(name+"/meta"):
        os.mkdir(name+"/meta")
    if not os.path.exists(name+"/img"):
        os.mkdir(name+"/img")
    for i in tqdm.tqdm(range(total_numbers)):
        meta = get_uri(base_uri, i)
        im = get_img(meta)
        output_uri(name, i, meta)
        output_image(name,i, im)

download(name, base_uri)