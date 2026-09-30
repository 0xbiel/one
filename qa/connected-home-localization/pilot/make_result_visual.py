from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import json,math
r=Path(__file__).parent;res=json.loads((r/'results_summary.json').read_text());scan=json.loads((r/'fixtures/connected_home/scan.json').read_text());walls=json.loads((r/'fixtures/connected_home/structural_boxes.json').read_text());im=Image.new('RGB',(1280,900),'#f3f5f7');d=ImageDraw.Draw(im);f=lambda n,b=False:ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold' if b else '')+'.ttf',n)
d.text((28,22),'Connected-home localization: measured results',font=f(28,True),fill='#162638');d.text((28,65),'24 synthetic tests • 12 distinct camera poses • calibrated + automatic focal length',font=f(18),fill='#475e72')
def xy(x,z):return(60+(x+7)*43,190+(z+5)*43)
for zone in scan['room_zones']:
 pts=[xy(p['x'],p['z']) for p in zone['polygon']];d.polygon(pts,fill='#e0e7ed' if zone['id']=='hallway' else '#ffffff');cx=sum(p[0] for p in pts)/4;cy=sum(p[1] for p in pts)/4;name=zone['id'].replace('_',' ');d.text((cx-55,cy-30),name,font=f(16,True),fill='#6a7b8a')
for b in walls:
 c=b['center'];dim=b['dimensions']
 if dim['y']<.5 or c['y']-dim['y']/2>.1:continue
 a=xy(c['x']-dim['x']/2,c['z']-dim['z']/2);q=xy(c['x']+dim['x']/2,c['z']+dim['z']/2);d.rectangle((*a,*q),fill='#344655')
for i,row in enumerate([x for x in res['rows'] if x['condition']=='clean']):
 if 'predicted_position' not in row:continue
 tx,_,tz=row['true_position'];px,_,pz=row['predicted_position'];a=xy(tx,tz);b=xy(px,pz);color='#c03737' if not row['loose_success'] else '#17836b'
 d.line((a,b),fill=color,width=3);d.ellipse((a[0]-5,a[1]-5,a[0]+5,a[1]+5),fill='#3277be');d.line((b[0]-5,b[1]-5,b[0]+5,b[1]+5),fill=color,width=2);d.line((b[0]-5,b[1]+5,b[0]+5,b[1]-5),fill=color,width=2)
d.text((55,136),'Calibrated-query map: blue = true camera; X = estimate',font=f(17),fill='#304c64');d.text((55,652),'Red: missed 25cm / 5° threshold. Lines show position error.',font=f(17),fill='#304c64')
s=res['all'];h=res['by_split']['heldout'];x=720;y=140
for title,value in [('Room identification',f"{s['room_correct']}/{s['n']} correct"),('Held-out room identification',f"{h['room_correct']}/{h['n']} correct"),('Within 10cm and 2°',f"{s['tight_success']}/{s['n']} cases"),('Within 25cm and 5°',f"{s['loose_success']}/{s['n']} cases"),('Median position / rotation',f"{s['position_error_m']['median']*100:.1f}cm / {s['rotation_error_deg']['median']:.2f}°"),('90th percentile position error',f"{s['position_error_m']['p90']:.2f}m")]:
 d.text((x,y),title,font=f(18),fill='#556c80');d.text((x,y+25),value,font=f(25,True),fill='#172d40');y+=85
d.text((28,745),'Key failure: confident placements can land in another room, up to 11.4m away.',font=f(23,True),fill='#a32e36');d.text((28,788),'Ideal simulated scan poses/depth. One home and camera model; these are regression results, not deployment accuracy.',font=f(17),fill='#475e72');d.text((28,821),'Actual ONE main 9c05bec worker + genuine YOLO-World detections. Query room/pose withheld from every request.',font=f(17),fill='#475e72');d.text((28,854),'All solver outcomes remain in denominators. Room assignment uses the final estimated camera center.',font=f(17),fill='#475e72');im.save(r/'localization_results.png')
