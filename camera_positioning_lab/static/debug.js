const $ = (id) => document.getElementById(id);
const mode = document.body.dataset.mode;
const state = {stream:null, timer:null, busy:false, running:false};

async function jfetch(url, options={}) {
  const response = await fetch(url, {headers:{'Content-Type':'application/json',...(options.headers||{})}, ...options});
  const text = await response.text();
  let body={}; try { body=text?JSON.parse(text):{}; } catch { body={detail:text}; }
  if(!response.ok) throw new Error(typeof body.detail==='string'?body.detail:JSON.stringify(body.detail||body));
  return body;
}
function log(kind,message){
  const row=document.createElement('div');row.className=`log-item ${kind}`;
  const time=document.createElement('span');time.textContent=new Date().toLocaleTimeString();
  const label=document.createElement('b');label.textContent=kind.toUpperCase();
  const detail=document.createElement('span');detail.textContent=message;
  row.append(time,label,detail);$('log').prepend(row);while($('log').children.length>100)$('log').lastElementChild.remove();
}
async function refreshStatus(){
  try{const s=await jfetch('/api/status');const w=s.worker||{};const ready=w.status==='ready';$('workerBadge').textContent=`Worker · ${w.status||'unknown'}${w.runtime?.device?` · ${w.runtime.device}`:w.device?` · ${w.device}`:''}`;$('workerBadge').className=`badge ${ready?'ready':'bad'}`;}
  catch{$('workerBadge').textContent='Worker · unavailable';$('workerBadge').className='badge bad';}
}
$('login').onclick=async()=>{
  try{$('login').disabled=true;const data=await jfetch('/api/login',{method:'POST',body:JSON.stringify({email:$('email').value.trim()})});const cameras=data.cameras||[];$('camera').replaceChildren();for(const camera of cameras){const option=document.createElement('option');option.value=camera.id;option.textContent=camera.name||camera.label||camera.id;$('camera').append(option);}if(!cameras.length){const option=document.createElement('option');option.textContent='No cameras';$('camera').append(option);} $('camera').disabled=!cameras.length;$('run').disabled=!state.stream||!cameras.length;log('ok',`Connected to ${data.me?.home?.name||data.home_id}; ${cameras.length} camera(s).`);}
  catch(e){log('fail',e.message);}finally{$('login').disabled=false;}
};
$('startCamera').onclick=async()=>{
  try{if(state.stream)state.stream.getTracks().forEach(t=>t.stop());state.stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720}},audio:false});$('video').srcObject=state.stream;await $('video').play();$('run').disabled=!$('camera').value;$('captureState').textContent='live';log('ok','Local camera opened.');}
  catch(e){log('fail',`Camera: ${e.message}`);}
};
function capture(){
  const video=$('video'),canvas=$('captureCanvas');const width=video.videoWidth,height=video.videoHeight;if(!width||!height)throw new Error('Camera is not producing frames yet.');canvas.width=width;canvas.height=height;canvas.getContext('2d').drawImage(video,0,0,width,height);return {frame_base64:canvas.toDataURL('image/jpeg',0.76).split(',')[1],width,height};
}
function drawFaces(data){
  const canvas=$('overlay'),video=$('video'),ctx=canvas.getContext('2d');canvas.width=video.clientWidth*devicePixelRatio;canvas.height=video.clientHeight*devicePixelRatio;ctx.clearRect(0,0,canvas.width,canvas.height);const sx=canvas.width/(video.videoWidth||1),sy=canvas.height/(video.videoHeight||1);
  for(const face of data.faces||[]){const box=face.bbox;if(!Array.isArray(box))continue;const matched=face.identity?.status==='matched';ctx.strokeStyle=matched?'#63e2ad':'#ffc95a';ctx.lineWidth=3*devicePixelRatio;ctx.strokeRect(box[0]*sx,box[1]*sy,(box[2]-box[0])*sx,(box[3]-box[1])*sy);ctx.fillStyle=matched?'#63e2ad':'#ffc95a';ctx.font=`${14*devicePixelRatio}px system-ui`;ctx.fillText(matched?face.identity.display_name||'Matched':'Unknown',box[0]*sx,Math.max(18*devicePixelRatio,box[1]*sy-7*devicePixelRatio));}
}
function drawFall(items){
  const canvas=$('overlay'),video=$('video'),ctx=canvas.getContext('2d');canvas.width=video.clientWidth*devicePixelRatio;canvas.height=video.clientHeight*devicePixelRatio;ctx.clearRect(0,0,canvas.width,canvas.height);const sx=canvas.width/(video.videoWidth||1),sy=canvas.height/(video.videoHeight||1);const edges=[[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
  for(const item of items){const b=item.bbox;if(!Array.isArray(b))continue;ctx.strokeStyle=item.gate?.status==='confirmed'?'#ff8396':'#58d3ff';ctx.lineWidth=2*devicePixelRatio;ctx.strokeRect(b[0]*sx,b[1]*sy,(b[2]-b[0])*sx,(b[3]-b[1])*sy);const pts=item.pose_keypoints;if(Array.isArray(pts)&&pts.length===17){ctx.strokeStyle='#aeeeff';ctx.lineWidth=2*devicePixelRatio;for(const [a,c] of edges){if(pts[a]?.[2]>.25&&pts[c]?.[2]>.25){ctx.beginPath();ctx.moveTo(pts[a][0]*sx,pts[a][1]*sy);ctx.lineTo(pts[c][0]*sx,pts[c][1]*sy);ctx.stroke();}}for(const p of pts){if(p?.[2]>.25){ctx.beginPath();ctx.arc(p[0]*sx,p[1]*sy,2.5*devicePixelRatio,0,Math.PI*2);ctx.fillStyle='#eafaff';ctx.fill();}}}}
}
function faceResults(data){
  const faces=data.faces||[];$('faceCount').textContent=`${faces.length} face${faces.length===1?'':'s'}`;$('results').replaceChildren();
  if(!faces.length){const p=document.createElement('p');p.className='muted';p.textContent=data.status==='blocked'?'Face processing is paused because this household has no active consented face-recognition profile.':data.status==='unavailable'?'Face model unavailable. Check model configuration and worker health.':'No face detected in this frame.';$('results').append(p);}
  for(const face of faces){const card=document.createElement('div');card.className='result-row';const identity=face.identity||{};const liveness=face.liveness||{};const heading=document.createElement('strong');heading.textContent=identity.status==='matched'?`Matched · ${identity.display_name}`:'Unknown person';const info=document.createElement('span');info.textContent=`Detection ${Math.round((face.confidence||0)*100)}% · Liveness ${liveness.status||'unavailable'}${liveness.score!=null?` (${Math.round(liveness.score*100)}%)`:''}`;const reason=document.createElement('small');reason.textContent=identity.reason|| (identity.status==='matched'?`Stable across ${identity.stability_hits} frames`:'No stable enrolled match');card.append(heading,info,reason);$('results').append(card);}
  drawFaces(data);
}
function fallResults(data){
  const items=data.diagnostics||[];$('personCount').textContent=`${items.length} tracked ${items.length===1?'person':'people'}`;$('results').replaceChildren();
  if(!items.length){const p=document.createElement('p');p.className='muted';p.textContent='No stable person track yet. Keep the person fully visible and wait a few seconds.';$('results').append(p);}
  for(const item of items){const card=document.createElement('div');card.className='result-row';const score=item.fall_sequence_probability;const gate=item.gate||{};const title=document.createElement('strong');title.textContent=item.would_emit?'Would emit · fall_suspected / needs_review':`Track ${item.track_id} · ${gate.status||'observing'}`;const scoreLine=document.createElement('span');scoreLine.textContent=`Sequence ${item.fall_sequence_status||'unknown'}${score!=null?` · ${Number(score).toFixed(3)} / threshold ${Number(item.fall_sequence_threshold).toFixed(2)} · ${item.fall_sequence_score_fresh?'fresh':'cached'}`:''}`;const gateLine=document.createElement('small');gateLine.textContent=`${gate.reason||'no gate reason'} · posture ${gate.posture||'unknown'} · keypoints ${gate.visible_keypoints??0}/17 · confirmation ${gate.confirmation_hits??0}/${gate.required_hits??3} hits, ${gate.confirmation_seconds??0}/${gate.required_seconds??0}s`;card.append(title,scoreLine,gateLine);$('results').append(card);}
  drawFall(items);
}
async function tick(){
  if(!state.running||state.busy||!state.stream||!$('camera').value)return;state.busy=true;
  try{$('captureState').textContent='processing';const frame=capture();const data=await jfetch(`/api/cameras/${encodeURIComponent($('camera').value)}/${mode==='faces'?'face':'fall'}-debug`,{method:'POST',body:JSON.stringify(frame)});if(mode==='faces'){faceResults(data);$('diagnostics').textContent=JSON.stringify({status:data.status,model_version:data.model_version,diagnostics:data.diagnostics,raw_frames_persisted:data.raw_frames_persisted,fall_scoring_ran:data.fall_scoring_ran},null,2);log('ok',`${(data.faces||[]).length} face(s); ${data.faces?.[0]?.identity?.status||'no face'}${data.faces?.[0]?.identity?.reason?` (${data.faces[0].identity.reason})`:''}`);}else{fallResults(data);$('diagnostics').textContent=JSON.stringify({status:data.status,detector_version:data.detector_version,diagnostics:data.diagnostics,raw_frames_persisted:data.raw_frames_persisted,observations_persisted:data.observations_persisted,events_persisted:data.events_persisted,debug_tracker_isolated:data.debug_tracker_isolated},null,2);log('ok',data.diagnostics?.length?data.diagnostics.map(x=>`${x.gate?.status||'observing'}: ${x.gate?.reason||'—'}${x.would_emit?' · would emit needs_review':''}`).join(' | '):'No stable person track yet.');}}
  catch(e){$('captureState').textContent='error';log('fail',e.message);}finally{state.busy=false;if(state.running)$('captureState').textContent='live';}
}
$('run').onclick=()=>{state.running=true;$('run').disabled=true;$('stop').disabled=false;$('captureState').textContent='starting';tick();state.timer=setInterval(tick,900);};
$('stop').onclick=()=>{state.running=false;clearInterval(state.timer);$('run').disabled=!state.stream;$('stop').disabled=true;$('captureState').textContent='stopped';};
$('clearLog').onclick=()=>{$('log').replaceChildren();};
window.addEventListener('beforeunload',()=>{clearInterval(state.timer);state.stream?.getTracks().forEach(track=>track.stop());});
refreshStatus();setInterval(refreshStatus,5000);
