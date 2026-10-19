"""Local-only figure galleries and half-size review sheets."""
import html
import json
import re
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def save_plot(fig,out,gallery,name,title,note):
    folder=out/'figures';folder.mkdir(parents=True,exist_ok=True)
    fig.suptitle(title,fontsize=13,fontweight='bold')
    fig.tight_layout(rect=[0,0,1,.95])
    fig.savefig(folder/(name+'.png'),dpi=120,bbox_inches='tight')
    plt.close(fig)
    gallery.append({'file':'figures/'+name+'.png','title':title,'note':note})


def write_gallery(out,gallery,title):
    (out/'gallery.json').write_text(json.dumps(gallery,ensure_ascii=False,indent=2),encoding='utf-8')
    cards=''.join(f'<section><h2>{html.escape(g["title"])}</h2><p>{html.escape(g["note"])}</p><a href="{g["file"]}"><img src="{g["file"]}" loading="lazy"></a></section>' for g in gallery)
    (out/'index.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{font:16px/1.6 'Malgun Gothic',sans-serif;background:#f4f5f5;max-width:1250px;margin:auto;padding:24px;color:#26333c}section{background:white;margin:24px 0;padding:20px;border:1px solid #ddd}img{width:100%}h2{font-size:20px}</style>'''+f'<title>{html.escape(title)}</title><h1>{html.escape(title)}</h1>'+cards+'</html>',encoding='utf-8')
    qa=out/'qa';qa.mkdir(exist_ok=True)
    for start in range(0,len(gallery),4):
        pics=[]
        for g in gallery[start:start+4]:
            with Image.open(out/g['file']) as im:
                im.load();pics.append(im.convert('RGB').resize((im.width//2,im.height//2),Image.Resampling.LANCZOS))
        width=max(im.width for im in pics);height=max(im.height for im in pics)+24
        canvas=Image.new('RGB',(width*2,height*2),'#ededed');pen=ImageDraw.Draw(canvas)
        for i,im in enumerate(pics):
            x=(i%2)*width;y=(i//2)*height
            canvas.paste(im,(x,y+24));pen.text((x+5,y+4),gallery[start+i]['file'],fill='black')
        canvas.save(qa/f'contact_{start//4+1:02d}.png')


def check_gallery(out,expected):
    gallery=json.loads((out/'gallery.json').read_text(encoding='utf-8'))
    assert len(gallery)==expected
    for g in gallery:
        with Image.open(out/g['file']) as im:im.verify()
    links=re.findall(r'(?:href|src)="([^"]+)"',(out/'index.html').read_text(encoding='utf-8'))
    assert len(links)==2*expected and all((out/link).is_file() for link in links)
    return {'figures':len(gallery),'gallery_links':len(links)}
