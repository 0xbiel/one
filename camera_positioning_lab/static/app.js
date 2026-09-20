const $ = (id) => document.getElementById(id);
const state = {
  stream:null,
  scene:null,
  candidate:null,
  candidateMatrix:null,
  candidateFov:null,
  candidateStatus:null,
  candidateSource:null,
  candidateConfidence:null,
  cameraId:null,
  continuous:false,
  continuousTimer:null,
  busy:false,
};

async function jfetch(url, options={}) {
  const response = await fetch(url, {headers:{'Content-Type':'application/json',...(options.headers||{})}, ...options});
  const text = await response.text();
  let body = {}; try { body = text ? JSON.parse(text) : {}; } catch { body = {detail:text}; }
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail || body));
  return body;
}

function log(kind, message) {
  const item=document.createElement('div'); item.className=`log-item ${kind}`;
  item.innerHTML=`<span>${new Date().toLocaleTimeString()}</span><b>${kind.toUpperCase()}</b><span></span>`;
  item.lastElementChild.textContent=message; $('log').prepend(item);
}

async function refreshStatus(){
  try { const s=await jfetch('/api/status'); const w=s.worker||{}; const ready=w.status==='ready'; $('workerBadge').textContent=`Worker · ${w.status||'unknown'}${w.runtime?.device?` · ${w.runtime.device}`:w.device?` · ${w.device}`:''}`; $('workerBadge').className=`badge ${ready?'ready':'bad'}`; }
  catch(e){ $('workerBadge').textContent='Worker · unavailable'; $('workerBadge').className='badge bad'; }
}

$('login').onclick=async()=>{
  try { $('login').disabled=true; const data=await jfetch('/api/login',{method:'POST',body:JSON.stringify({email:$('email').value.trim()})}); const cams=data.cameras||[]; $('camera').innerHTML=cams.length?cams.map(c=>`<option value="${c.id}">${c.name||c.label||c.id}${c.room_id?` · ${c.room_id}`:''}</option>`).join(''):'<option>No cameras</option>'; $('camera').disabled=!cams.length; $('refresh').disabled=!cams.length; state.cameraId=cams[0]?.id||null; log('ok',`Connected to ${data.me?.home?.name||data.home_id}; ${cams.length} camera(s).`); if(state.cameraId) await loadRoom(); }
  catch(e){ log('fail',e.message); }
  finally{$('login').disabled=false;}
};
$('camera').onchange=async()=>{
  state.cameraId=$('camera').value;
  state.candidate=null;
  state.candidateMatrix=null;
  state.candidateFov=null;
  state.candidateStatus=null;
  state.candidateSource=null;
  state.candidateConfidence=null;
  await loadRoom();
};
$('refresh').onclick=loadRoom;

function zonesFromScene(scene){ return scene?.geometry?.room_zones || scene?.geometry?.roomZones || []; }
function registrations(scene){ return scene?.cameraRegistrations || scene?.camera_registrations || (scene?.cameraRegistration?[scene.cameraRegistration]:[]); }
function roomObjects(scene){ return scene?.canonicalGeometry?.objects || []; }
function roomDoors(scene){ return scene?.canonicalGeometry?.doors || []; }
function centerOfMatrix(m){ return Array.isArray(m)&&m.length===4?[Number(m[0][3]),Number(m[1][3]),Number(m[2][3])]:null; }
function validMatrix(m){ return Array.isArray(m)&&m.length===4&&m.every(row=>Array.isArray(row)&&row.length===4&&row.every(value=>Number.isFinite(Number(value)))); }
function finiteFov(value){ const fov=Number(value); return Number.isFinite(fov)&&fov>=30&&fov<=120?fov:null; }
function registrationFov(registration){
  const explicit=finiteFov(registration?.intrinsics?.fov_degrees ?? registration?.intrinsics?.fovDegrees ?? registration?.metrics?.diagnostics?.selected_fov_degrees);
  if(explicit)return explicit;
  const matrix=registration?.intrinsics?.matrix;
  if(!Array.isArray(matrix)||matrix.length!==3||matrix.some(row=>!Array.isArray(row)||row.length!==3))return null;
  const fx=Number(matrix[0][0]),cx=Number(matrix[0][2]);
  if(!Number.isFinite(fx)||fx<=0||!Number.isFinite(cx)||cx<=0)return null;
  return finiteFov(2*Math.atan(cx/fx)*180/Math.PI);
}

function polygonArea(points){
  if(!Array.isArray(points)||points.length<3)return 0;
  let sum=0;
  for(let i=0;i<points.length;i++){
    const a=points[i],b=points[(i+1)%points.length];
    sum+=Number(a.x)*Number(b.z)-Number(b.x)*Number(a.z);
  }
  return Math.abs(sum)*.5;
}

function pointInPolygon(point,polygon){
  let inside=false;
  for(let i=0,j=polygon.length-1;i<polygon.length;j=i++){
    const a=polygon[i],b=polygon[j];
    const ax=Number(a.x),az=Number(a.z),bx=Number(b.x),bz=Number(b.z);
    const crosses=((az>point.z)!==(bz>point.z))&&(point.x<(bx-ax)*(point.z-az)/(bz-az)+ax);
    if(crosses)inside=!inside;
  }
  return inside;
}

function distanceToSegment(point,a,b){
  const dx=Number(b.x)-Number(a.x),dz=Number(b.z)-Number(a.z);
  const length2=dx*dx+dz*dz;
  if(length2<=1e-9)return Math.hypot(point.x-Number(a.x),point.z-Number(a.z));
  const t=Math.max(0,Math.min(1,((point.x-Number(a.x))*dx+(point.z-Number(a.z))*dz)/length2));
  return Math.hypot(point.x-(Number(a.x)+t*dx),point.z-(Number(a.z)+t*dz));
}

function zoneForPoint(point,zones){
  const containing=zones.find(zone=>pointInPolygon(point,zone.polygon));
  if(containing)return containing;
  let best=null,bestDistance=Infinity;
  zones.forEach(zone=>{
    for(let i=0;i<zone.polygon.length;i++){
      const distance=distanceToSegment(point,zone.polygon[i],zone.polygon[(i+1)%zone.polygon.length]);
      if(distance<bestDistance){bestDistance=distance;best=zone;}
    }
  });
  return bestDistance<=0.4?best:null;
}

function raySegmentDistance(origin,direction,a,b){
  const sx=Number(b.x)-Number(a.x),sz=Number(b.z)-Number(a.z);
  const cross=direction.x*sz-direction.z*sx;
  if(Math.abs(cross)<1e-8)return null;
  const qx=Number(a.x)-origin.x,qz=Number(a.z)-origin.z;
  const t=(qx*sz-qz*sx)/cross;
  const u=(qx*direction.z-qz*direction.x)/cross;
  return t>1e-4&&u>=-1e-6&&u<=1+1e-6?t:null;
}

function rayToPolygonDistance(origin,direction,polygon,maxDistance){
  let nearest=maxDistance;
  for(let i=0;i<polygon.length;i++){
    const distance=raySegmentDistance(origin,direction,polygon[i],polygon[(i+1)%polygon.length]);
    if(distance!==null&&distance<nearest)nearest=distance;
  }
  return nearest;
}

function cameraGroundRay(matrix,angleRadians){
  const localX=Math.sin(angleRadians),localZ=-Math.cos(angleRadians);
  let x=Number(matrix[0][0])*localX+Number(matrix[0][2])*localZ;
  let z=Number(matrix[2][0])*localX+Number(matrix[2][2])*localZ;
  const length=Math.hypot(x,z);
  if(length<1e-6)return null;
  x/=length;z/=length;
  return {x,z};
}

function fovFootprint(matrix,fov,zones,maxDistance,samples=31){
  if(!validMatrix(matrix)||!finiteFov(fov))return null;
  const center=centerOfMatrix(matrix);if(!center)return null;
  const origin={x:center[0],z:center[2]};
  const zone=zoneForPoint(origin,zones);
  const boundary=zone?.polygon||null;
  const endpoints=[];
  const half=Number(fov)*Math.PI/360;
  for(let i=0;i<samples;i++){
    const angle=-half+(i/(samples-1))*half*2;
    const direction=cameraGroundRay(matrix,angle);if(!direction)continue;
    const distance=boundary?rayToPolygonDistance(origin,direction,boundary,maxDistance):maxDistance;
    endpoints.push({x:origin.x+direction.x*distance,z:origin.z+direction.z*distance});
  }
  if(endpoints.length<2)return null;
  return {origin,zone,points:[origin,...endpoints],endpoints};
}

function drawFovCone(ctx,matrix,fov,zones,distance,tx,tz,{fill,stroke,label}){
  const footprint=fovFootprint(matrix,fov,zones,distance);
  if(!footprint)return null;
  const middle=footprint.endpoints[Math.floor(footprint.endpoints.length/2)];
  ctx.save();
  ctx.beginPath();footprint.points.forEach((point,index)=>{index?ctx.lineTo(tx(point.x),tz(point.z)):ctx.moveTo(tx(point.x),tz(point.z));});ctx.closePath();
  ctx.fillStyle=fill;ctx.fill();ctx.strokeStyle=stroke;ctx.lineWidth=1.5;ctx.stroke();
  ctx.beginPath();ctx.moveTo(tx(footprint.origin.x),tz(footprint.origin.z));ctx.lineTo(tx(middle.x),tz(middle.z));ctx.strokeStyle=stroke;ctx.setLineDash([5,4]);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle=label;ctx.font='11px system-ui';ctx.fillText(`${Number(fov).toFixed(0)}° FOV`,tx(middle.x)+6,tz(middle.z)-5);
  ctx.restore();
  return footprint;
}

function transformedPlanCorners(item){
  const matrix=item?.transform,dimensions=item?.dimensions;
  if(!validMatrix(matrix)||!dimensions)return [];
  const hx=Number(dimensions.x)/2,hz=Number(dimensions.z)/2;
  if(!Number.isFinite(hx)||!Number.isFinite(hz))return [];
  return [[-hx,-hz],[hx,-hz],[hx,hz],[-hx,hz]].map(([x,z])=>({
    x:Number(matrix[0][0])*x+Number(matrix[0][2])*z+Number(matrix[0][3]),
    z:Number(matrix[2][0])*x+Number(matrix[2][2])*z+Number(matrix[2][3]),
  }));
}

function drawRoomObjects(ctx,scene,tx,tz){
  roomObjects(scene).forEach(item=>{
    const corners=transformedPlanCorners(item);if(corners.length!==4)return;
    ctx.save();ctx.beginPath();corners.forEach((point,index)=>{index?ctx.lineTo(tx(point.x),tz(point.z)):ctx.moveTo(tx(point.x),tz(point.z));});ctx.closePath();
    ctx.fillStyle='rgba(145,184,205,.16)';ctx.fill();ctx.strokeStyle='rgba(145,184,205,.62)';ctx.lineWidth=1.2;ctx.stroke();
    const center=item.center||{x:Number(item.transform?.[0]?.[3]),z:Number(item.transform?.[2]?.[3])};
    if(Number.isFinite(Number(center.x))&&Number.isFinite(Number(center.z))){ctx.fillStyle='#9bbdcd';ctx.font='10px system-ui';ctx.fillText(item.category||item.label||'object',tx(Number(center.x))+4,tz(Number(center.z))-4);}
    ctx.restore();
  });
  roomDoors(scene).forEach(item=>{
    const matrix=item?.transform,dimensions=item?.dimensions;if(!validMatrix(matrix)||!dimensions)return;
    const half=Number(dimensions.x)/2;if(!Number.isFinite(half))return;
    const point=(x)=>({x:Number(matrix[0][0])*x+Number(matrix[0][3]),z:Number(matrix[2][0])*x+Number(matrix[2][3])});
    const a=point(-half),b=point(half);ctx.save();ctx.strokeStyle='#77e1bd';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(tx(a.x),tz(a.z));ctx.lineTo(tx(b.x),tz(b.z));ctx.stroke();ctx.restore();
  });
}

function drawMap(){
  const c=$('map'), ctx=c.getContext('2d'); ctx.clearRect(0,0,c.width,c.height); ctx.fillStyle='#071119';ctx.fillRect(0,0,c.width,c.height);
  const zones=zonesFromScene(state.scene).filter(z=>Array.isArray(z.polygon)&&z.polygon.length>=3);
  const pts=zones.flatMap(z=>z.polygon.map(p=>({x:Number(p.x),z:Number(p.z)}))).filter(p=>Number.isFinite(p.x)&&Number.isFinite(p.z));
  if(!pts.length){ctx.fillStyle='#7898a8';ctx.font='15px system-ui';ctx.fillText('No RoomPlan floor polygon in scene',24,34);return;}
  const xs=pts.map(p=>p.x), zs=pts.map(p=>p.z), minX=Math.min(...xs),maxX=Math.max(...xs),minZ=Math.min(...zs),maxZ=Math.max(...zs); const pad=48; const sx=(c.width-pad*2)/Math.max(.1,maxX-minX), sz=(c.height-pad*2)/Math.max(.1,maxZ-minZ), scale=Math.min(sx,sz); const tx=x=>pad+(x-minX)*scale, tz=z=>c.height-pad-(z-minZ)*scale; const viewDistance=Math.max(2,Math.hypot(maxX-minX,maxZ-minZ)*1.25);
  zones.forEach((z,i)=>{ctx.beginPath();z.polygon.forEach((p,j)=>{const x=tx(Number(p.x)),y=tz(Number(p.z));j?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.closePath();ctx.fillStyle='rgba(59,153,190,.11)';ctx.fill();ctx.strokeStyle='#2b6985';ctx.lineWidth=2;ctx.stroke();ctx.fillStyle='#8eb9cc';ctx.font='12px system-ui';const p=z.polygon[0];ctx.fillText(z.label||z.id||`room ${i+1}`,tx(Number(p.x))+7,tz(Number(p.z))-7)});
  const saved=[];
  const palette=[['rgba(82,213,255,.13)','rgba(82,213,255,.78)','#9bdfff','#52d5ff'],['rgba(164,140,255,.13)','rgba(164,140,255,.78)','#c9bbff','#a48cff'],['rgba(61,197,149,.13)','rgba(61,197,149,.78)','#9ff1d2','#3dc595']];
  registrations(state.scene).filter(r=>r.status==='positioned').forEach((r,index)=>{
    const m=r.cameraToWorld||r.camera_to_world,fov=registrationFov(r),colors=palette[index%palette.length];
    const footprint=drawFovCone(ctx,m,fov,zones,viewDistance,tx,tz,{fill:colors[0],stroke:colors[1],label:colors[2]});
    const p=centerOfMatrix(m);if(!p)return;ctx.fillStyle=colors[3];ctx.beginPath();ctx.arc(tx(p[0]),tz(p[2]),6,0,Math.PI*2);ctx.fill();ctx.fillStyle=colors[2];ctx.fillText(r.cameraName||r.cameraId||'camera',tx(p[0])+10,tz(p[2])-7);
    const roomArea=footprint?.zone?polygonArea(footprint.zone.polygon):0,coveredArea=footprint?polygonArea(footprint.points):0;
    saved.push({name:r.cameraName||r.cameraId||'camera',fov,coverage:roomArea>0?Math.min(100,coveredArea/roomArea*100):null});
  });
  if(state.candidate){
    const rejected=state.candidateStatus&&state.candidateStatus!=='positioned';
    const footprint=drawFovCone(ctx,state.candidateMatrix,state.candidateFov,zones,viewDistance,tx,tz,{fill:rejected?'rgba(255,143,164,.08)':'rgba(255,201,90,.15)',stroke:rejected?'rgba(255,143,164,.86)':'rgba(255,201,90,.82)',label:rejected?'#ff9cad':'#ffe19a'});
    ctx.fillStyle='#ffc95a';ctx.beginPath();ctx.arc(tx(state.candidate[0]),tz(state.candidate[2]),8,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#fff1bd';ctx.lineWidth=2;ctx.stroke();ctx.fillStyle='#ffe19a';ctx.fillText(`candidate ${state.candidate[0].toFixed(2)}, ${state.candidate[2].toFixed(2)}`,tx(state.candidate[0])+12,tz(state.candidate[2])+4);
    if(footprint&&rejected){ctx.fillStyle='#ff9cad';ctx.font='10px system-ui';ctx.fillText('diagnostic only',tx(state.candidate[0])+12,tz(state.candidate[2])+18);}
  }
  drawRoomObjects(ctx,state.scene,tx,tz);
  const candidateParts=[];
  if(state.candidateFov)candidateParts.push(`${state.candidateFov.toFixed(0)}° FOV`);
  if(state.candidateSource)candidateParts.push(state.candidateSource);
  if(state.candidateConfidence!=null)candidateParts.push(`${Math.round(state.candidateConfidence*100)}% confidence`);
  if(state.candidateStatus)candidateParts.push(state.candidateStatus==='positioned'?'pose accepted':'pose rejected');
  const savedText=saved.length?saved.map(item=>`${item.name}: ${item.fov?`${item.fov.toFixed(0)}°`: 'FOV unavailable'}${item.coverage!=null?` · ${item.coverage.toFixed(0)}% room footprint`:''}`).join('  ·  '):'No accepted camera pose saved for this map';
  $('mapMeta').textContent=`${savedText}${candidateParts.length?`  |  Current estimate: ${candidateParts.join(' · ')}`:''}`;
}

async function loadRoom(){
  if(!state.cameraId)return;
  try { const [ready,scene]=await Promise.all([jfetch(`/api/cameras/${state.cameraId}/readiness`),jfetch(`/api/cameras/${state.cameraId}/scene`)]); state.scene=scene; $('roomLabel').textContent=`map ${ready.map_id?.slice(0,8)||'—'} · landmarks ${ready.visual_landmarks_ready?'ready':'not ready'}`; $('diagnostics').textContent=JSON.stringify({readiness:ready},null,2); drawMap(); log(ready.ready?'ok':'warn',`RoomPlan readiness: ${ready.ready?'ready':'not ready'}.`); }
  catch(e){log('fail',`Room load: ${e.message}`);}
}

$('startCamera').onclick=async()=>{
  try { if(state.stream) state.stream.getTracks().forEach(t=>t.stop()); state.stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720}},audio:false}); $('video').srcObject=state.stream; $('run').disabled=false; $('continuous').disabled=false; $('captureState').textContent='live'; log('ok','Mac camera opened.'); }
  catch(e){log('fail',`Camera: ${e.message}`);}
};

async function captureFrame(){
  const v=$('video'), c=$('captureCanvas'); const w=v.videoWidth||1280,h=v.videoHeight||720;c.width=w;c.height=h;const ctx=c.getContext('2d');ctx.drawImage(v,0,0,w,h);const data=c.toDataURL('image/jpeg',0.76);return {frame_base64:data.split(',')[1],width:w,height:h};
}
async function captureBurst(count=6,duration=1500){const result=[];for(let i=0;i<count;i++){result.push(await captureFrame());if(i<count-1)await new Promise(r=>setTimeout(r,duration/(count-1)));}return result;}

async function runLocalization(){
  if(state.busy||!state.stream||!state.cameraId)return;state.busy=true;$('run').disabled=true;$('captureState').textContent='capturing';const started=performance.now();
  try { const frames=await captureBurst(); $('captureState').textContent='solving'; const rawFov=$('fov').value.trim(); const request={frames}; if(rawFov)request.fov_degrees=Number(rawFov); const result=await jfetch(`/api/cameras/${state.cameraId}/localize`,{method:'POST',body:JSON.stringify(request)}); const d=result.diagnostics||{}; const matrix=result.camera_to_world; const diagnosticMatrix=d.selected_candidate_camera_to_world; state.candidate=centerOfMatrix(matrix)||d.selected_camera_center||d.geometric_selected_camera_center||null; state.candidateMatrix=validMatrix(matrix)?matrix:(validMatrix(diagnosticMatrix)?diagnosticMatrix:null); state.candidateFov=finiteFov(d.selected_fov_degrees ?? (rawFov||null)); state.candidateStatus=result.status||null; state.candidateSource=d.selected_estimate_source||null; const selectedConfidence=d.selected_estimate_confidence ?? result.confidence; state.candidateConfidence=Number.isFinite(Number(selectedConfidence))?Number(selectedConfidence):null; $('status').textContent=result.status||'—'; $('inliers').textContent=`${result.inlier_count??0} / ${result.match_count??0}`; $('reprojection').textContent=result.reprojection_error_px==null?'—':`${Number(result.reprojection_error_px).toFixed(2)} px`; $('elapsed').textContent=`${((performance.now()-started)/1000).toFixed(2)} s`; $('diagnostics').textContent=JSON.stringify(result,null,2); drawMap(); const center=state.candidate?` center=[${state.candidate.map(v=>Number(v).toFixed(2)).join(', ')}]`:''; const fov=state.candidateFov?` fov=${state.candidateFov.toFixed(0)}°`:''; log(result.status==='positioned'?'ok':'warn',`${result.status}; inliers ${result.inlier_count??0}; error ${result.reprojection_error_px??'—'}${fov}${center}`); }
  catch(e){$('status').textContent='error';log('fail',e.message);}
  finally{state.busy=false;$('run').disabled=false;$('captureState').textContent='live';}
}
$('run').onclick=runLocalization;
$('continuous').onclick=()=>{state.continuous=!state.continuous;$('continuous').textContent=state.continuous?'Stop continuous':'Start continuous';if(state.continuous){runLocalization();state.continuousTimer=setInterval(runLocalization,6500)}else{clearInterval(state.continuousTimer)}};
$('clearLog').onclick=()=>{$('log').innerHTML=''};
refreshStatus();setInterval(refreshStatus,4000);
