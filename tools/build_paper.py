"""Build the PDF from Markdown and frozen numeric results; never fit a model."""
from pathlib import Path
from xml.sax.saxutils import escape
import json
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,LongTable,TableStyle,Image,KeepTogether,PageBreak
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'paper'
data=json.loads((ROOT/'evidence/study015/report.json').read_text())
rows={r['name']:r for r in data['strategies']}
arms=[r['arm'] for r in data['strategies'] if r['timeframe']=='2h']
names=['Alpha','Current','Compact8','Robust8','T-mean','T-guard','Ridge8','X-current','X-compact','X-robust']
plt.rcParams.update({'font.size':8.5,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(7.1,8.3),constrained_layout=True)
for i,tf in enumerate(['2h','4h']):
    for j,period in enumerate(['development','recent']):
        ax=axes[i,j]; values=[rows[tf+'_'+a][period]['base']['return']*100 for a in arms]
        bars=ax.barh(range(10),values,color=['#83929e','#183e62']+['#35836e']*5+['#ae774b']*3,height=.7)
        ax.set_yticks(range(10),names); ax.invert_yaxis(); ax.axvline(0,color='#333333',lw=.6)
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=7.8)
        lo,hi=min(0,min(values)),max(0,max(values)); span=max(5,hi-lo)
        ax.set_xlim(lo-.28*span,hi+.32*span)
        ax.set_title(tf+' / '+('2024-2025' if period=='development' else 'known 2026'),fontsize=10)
        ax.set_xlabel('Net return (%)',fontsize=8)
        ax.tick_params(axis='both',labelsize=8)
fig.savefig(PAPER/'figures/comparison.png',dpi=210)
plt.close(fig)

for name,family,weight in [('Body','DejaVu Serif','normal'),('BodyBold','DejaVu Serif','bold'),('Sans','DejaVu Sans','normal'),('SansBold','DejaVu Sans','bold')]:
    path=font_manager.findfont(font_manager.FontProperties(family=family,weight=weight))
    pdfmetrics.registerFont(TTFont(name,path))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='Body',boldItalic='BodyBold')
pdfmetrics.registerFontFamily('Sans',normal='Sans',bold='SansBold',italic='Sans',boldItalic='SansBold')
styles={
    'title':ParagraphStyle('title',fontName='SansBold',fontSize=21,leading=25,spaceAfter=14,textColor=colors.HexColor('#173b57')),
    'body':ParagraphStyle('body',fontName='Body',fontSize=10.1,leading=14.1,alignment=TA_JUSTIFY,spaceAfter=8),
    'meta':ParagraphStyle('meta',fontName='Sans',fontSize=8.8,leading=12,spaceAfter=12),
    'h2':ParagraphStyle('h2',fontName='SansBold',fontSize=13.2,leading=17,spaceBefore=12,spaceAfter=7,keepWithNext=True,textColor=colors.HexColor('#173b57')),
    'h3':ParagraphStyle('h3',fontName='SansBold',fontSize=10.8,leading=14,spaceBefore=8,spaceAfter=5,keepWithNext=True),
    'cell':ParagraphStyle('cell',fontName='Sans',fontSize=8,leading=10.4),
    'header':ParagraphStyle('header',fontName='SansBold',fontSize=8,leading=10.4,textColor=colors.white),
    'caption':ParagraphStyle('caption',fontName='Sans',fontSize=8.4,leading=11.5,spaceBefore=5,spaceAfter=10),
    'ref':ParagraphStyle('ref',fontName='Body',fontSize=8.7,leading=12,spaceAfter=7),
}
def inline(value):
    value=escape(value.replace('\u2013','-').replace('\u2014','-').replace('\u2011','-'))
    value=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',value)
    value=re.sub(r'`([^`]+)`',lambda m:'<font name="Sans" size="9">'+m[1]+'</font>',value)
    value=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2" color="#174966">\1</a>',value)
    return value

lines=(PAPER/'manuscript.md').read_text(encoding='utf-8').splitlines()
story=[]; i=0; references=False; width=A4[0]-100
while i<len(lines):
    line=lines[i].strip()
    if not line: i+=1; continue
    if line.startswith('# '): story.append(Paragraph(inline(line[2:]),styles['title'])); i+=1; continue
    if line.startswith('## '):
        references=line=='## References'
        if line.startswith('## Appendix C.'):
            story.append(PageBreak())
        story.append(Paragraph(inline(line[3:]),styles['h2'])); i+=1; continue
    if line.startswith('### '): story.append(Paragraph(inline(line[4:]),styles['h3'])); i+=1; continue
    if line.startswith('|'):
        table=[]
        while i<len(lines) and lines[i].strip().startswith('|'):
            cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r'[:\- ]+',c) for c in cells): table.append(cells)
            i+=1
        n=len(table[0]); widths=([91,width-191,100] if n==3 else [26,145,58,(width-229)/2,(width-229)/2] if n==5 else [26,117]+[(width-143)/4]*4)
        values=[[Paragraph(inline(c),styles['header' if row==0 else 'cell']) for c in cells] for row,cells in enumerate(table)]
        t=LongTable(values,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#173b57')),('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f0f4f6')]),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#173b57'))]))
        story.append(t); story.append(Spacer(1,4)); continue
    if line.startswith('!['):
        img=Image(str(PAPER/'figures/comparison.png'),width=width,height=width*8.3/7.1)
        i+=1
        while i<len(lines) and not lines[i].strip(): i+=1
        caption=lines[i].strip() if i<len(lines) and lines[i].startswith('Figure ') else ''
        if caption: i+=1
        story.append(KeepTogether([img,Paragraph(inline(caption+' T denotes temporal; X denotes opposite-asset training.'),styles['caption'])])); continue
    paragraph=[line]; i+=1
    while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![')):
        if references and re.match(r'^\d+\.\s',lines[i]): break
        paragraph.append(lines[i].strip()); i+=1
    text=' '.join(paragraph)
    kind='ref' if references else 'caption' if text.startswith(('Table ','Figure ')) else 'meta' if text.startswith('**Author:') else 'body'
    rendered='<br/>'.join(inline(v) for v in paragraph) if kind=='meta' else inline(text)
    story.append(Paragraph(rendered,styles[kind]))

def page(canvas,doc):
    canvas.saveState(); w,h=A4
    canvas.setFont('Sans',7.5); canvas.setFillColor(colors.HexColor('#5b6973'))
    canvas.drawString(50,31,'wizcap / AlphaTrend generalization / preprint v1.0.0')
    canvas.drawRightString(w-50,31,str(doc.page))
    if doc.page>1:
        canvas.drawString(50,h-28,'RESEARCH PREPRINT - NOT PEER REVIEWED')
        canvas.setStrokeColor(colors.HexColor('#d2dbe1')); canvas.line(50,h-35,w-50,h-35)
    canvas.restoreState()

dest=PAPER/'alphatrend-generalization.pdf'
doc=SimpleDocTemplate(str(dest),pagesize=A4,leftMargin=50,rightMargin=50,topMargin=48,bottomMargin=48,
    title='Generalization Limits of Adaptive AlphaTrend Exit Models in Cryptocurrency Futures',author='wizcap',subject='Frozen discovery-only cryptocurrency futures research',pageCompression=1)
doc.build(story,onFirstPage=page,onLaterPages=page)
reader=PdfReader(dest)
assert all(p.extract_text().strip() for p in reader.pages)
assert '28.73' in ''.join(p.extract_text() for p in reader.pages)
print('Built',dest.name,'pages',len(reader.pages))
