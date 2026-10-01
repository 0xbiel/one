import json,pathlib
from PIL import Image,ImageDraw,ImageFont
R=pathlib.Path(__file__).parent
J=json.load(open(R/'full_scores.json'));C=json.load(open(R/'cadence20_scores.json'))
font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';bold='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf';F=lambda n,b=False:ImageFont.truetype(bold if b else font,n)
im=Image.new('RGB',(1600,1100),'#101b28');d=ImageDraw.Draw(im)
d.text((55,35),'ONE  /  REAL FALL VIDEO INVESTIGATION',font=F(38,True),fill='white')
d.text((55,92),'160 staged clips  •  6,439 RGB samples  •  128 evaluation clips',font=F(25),fill='#a7c0d5')
d.text((55,139),'Evaluation: 63 fall clips + 65 normal-activity clips',font=F(23),fill='#a7c0d5')
d.text((55,197),'SAME FALL RULE, DIFFERENT SAMPLING / DETECTOR',font=F(24,True),fill='#68d5c1')
rows=[]
for method,label,hz in [('world_baseline','ONE, 1 frame / 1.4 s',1/1.4),('world_baseline','ONE, 5 frames / s',5),('pose_box_baseline','Pose detector boxes, 5 frames / s',5)]:
 r=next(x for x in J['summary'] if x['split']=='evaluation' and x['method']==method and abs(x['hz']-hz)<1e-5);rows.append((label,r))
for i,(label,r) in enumerate(rows):
 y=250+i*124;d.text((55,y),label,font=F(25,True),fill='white')
 d.rounded_rectangle((55,y+43,820,y+76),radius=10,fill='#263d50')
 if r['TP']:d.rounded_rectangle((55,y+43,55+765*r['TP']/63,y+76),radius=10,fill='#55c7b0')
 d.text((850,y+33),f"{r['TP']} / 63 falls detected",font=F(27,True),fill='#55c7b0')
 d.text((850,y+72),f"{r['FP']} / 65 controls false-alerted",font=F(23),fill='#ffb879')
d.line((55,630,1545,630),fill='#344b5e',width=2)
d.text((55,665),'20 FPS: PAIRED DIFFICULT SUBSET',font=F(26,True),fill='#68d5c1')
for i,hz in enumerate([5,20]):
 r=next(x for x in C['summary'] if x['split']=='evaluation' and x['method']=='world_baseline' and x['hz']==hz)
 d.text((55+i*760,718),f"{hz} fps: {r['TP']}/4 falls; {r['FP']}/5 controls false-alerted",font=F(25,True),fill='white')
d.text((55,770),'Higher input rate alone did not resolve these failure modes',font=F(26),fill='#c0d1de')
d.rounded_rectangle((45,835,1555,1035),radius=15,fill='#192b3c')
lines=['Offline replay, not a deployed safety-system validation. Tracker resets for every clip.',
'Low-rate results are affected by short pre-fall context. Sample phase also matters.',
'Pose comparison uses its boxes through the same rule, not a trained skeleton classifier.',
'GMDCSA-24 (Alam et al., 2024), MIT repository license. See report for source and limits.']
for i,l in enumerate(lines):d.text((65,860+i*39),l,font=F(21),fill='#a7c0d5')
im.save(R/'ONE-real-fall-benchmark-summary.png');print('summary image created')
