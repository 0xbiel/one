import pathlib,subprocess,json,sys
R=pathlib.Path(__file__).parent;prefix=sys.argv[1] if len(sys.argv)>1 else 'cadence20_';fps=20 if prefix=='cadence20_' else 5;manifest='manifest20.json' if fps==20 else 'full_manifest_downloaded.json';period=round(fps*1.4);allrows=[]
for phase in range(period):
 subprocess.run([sys.executable,str(R/'score.py'),'--prefix',prefix,'--manifest',manifest,'--base-fps',str(fps),'--rates','0.7142857142857143','--phase',str(phase)],stdout=subprocess.DEVNULL,check=True)
 f=R/f'{prefix}scores'+pathlib.Path('') if False else R/f'{prefix}scores{"_phase"+str(phase) if phase else ""}.json'
 j=json.load(open(f))
 for x in j['summary']:allrows.append(dict(**x,phase_offset_s=phase/fps))
json.dump(allrows,open(R/f'{prefix}phase_sensitivity.json','w'),indent=2)
# Restore primary fullrate report after phase0 call.
rates='20,5,0.7142857142857143' if fps==20 else '5,0.7142857142857143'
subprocess.run([sys.executable,str(R/'score.py'),'--prefix',prefix,'--manifest',manifest,'--base-fps',str(fps),'--rates',rates],stdout=subprocess.DEVNULL,check=True)
print('phase sweep finished',prefix,period)
