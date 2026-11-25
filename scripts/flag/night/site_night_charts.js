const NT=__NIGHT_DATA__;
const NMK={sprobe:{n:'Σ·probe',m:{size:13,color:HU,line:{width:1,color:'#fff'}}}, fitv:{n:'fitted v',m:{size:13,color:AI,line:{width:1,color:'#fff'}}},
  gf:{n:'Goodfire',m:{size:12,symbol:'diamond',color:INK,line:{width:1,color:'#fff'}}}, noise:{n:'noise-only',m:{size:11,symbol:'square-open',color:AI,line:{width:2}}}};
const ISO={size:8,color:GR,opacity:0.5}, SIG={size:8,symbol:'circle-open',color:GR,line:{width:1.8}};

// detection on single-turn environments: probe (left out) against Goodfire's direction
(function(){const d=NT.det, E=Object.keys(d.gf_pool).sort((a,b)=>d.gf_pool[a][0]-d.gf_pool[b][0]);
  const S=[['probe','logistic probe, environment left out',{size:11,color:AI},-0.12,AI],['gf_pool','Goodfire direction',{size:11,symbol:'diamond',color:INK},0.12,INK]];
  const tr=S.map(([k,n,mk,dx,col])=>({x:E.map((_,i)=>i+dx),y:E.map(e=>d[k][e][0]),error_y:{type:'data',symmetric:false,array:E.map(e=>d[k][e][2]-d[k][e][0]),arrayminus:E.map(e=>d[k][e][0]-d[k][e][1]),color:col,thickness:1.2,width:0},mode:'markers',marker:mk,hovertemplate:n+': %{y:.2f}<extra></extra>'}));
  Plotly.newPlot('n-det',tr,L({height:340,shapes:[{type:'line',x0:-0.5,x1:E.length-0.5,y0:0.5,y1:0.5,line:{color:'#cfc4b4',width:1,dash:'dot'}}],margin:{l:56,r:10,t:10,b:86},
    xaxis:ax('',{tickvals:E.map((_,i)=>i),ticktext:E.map(e=>e.replace('_','<br>')),gridcolor:'rgba(0,0,0,0)',range:[-0.5,E.length-0.5],tickfont:{size:11.5}}),yaxis:ax('matched AUROC',{range:[-0.03,1.04]})}),cfg);})();

// subtraction against the random null
(function(){const tr=[]; const rows=[{name:'Five held-out<br>environments',base:NT.base,iso:NT.rnd_sub.iso.concat(NT.rnd_add.iso),sig:NT.rnd_sub.sig.concat(NT.rnd_add.sig),sub:NT.sub},
    {name:'Three fitting<br>environments',base:NT.fit.base,iso:NT.fit.rnd,sig:[],sub:NT.fit.sub}];
  rows.forEach((g,i)=>{
    tr.push({x:g.iso.map(d=>g.base+d),y:g.iso.map((_,j)=>i-0.12+((j%4)-1.5)*0.05),mode:'markers',marker:ISO,hovertemplate:'random'+(i?'':' (isotropic)')+': %{x:.0f}%<extra></extra>'});
    tr.push({x:g.sig.map(d=>g.base+d),y:g.sig.map((_,j)=>i+0.14+((j%4)-1.5)*0.05),mode:'markers',marker:SIG,hovertemplate:'random (Σ-shaped): %{x:.0f}%<extra></extra>'});
    tr.push({x:[g.base,g.base],y:[i-0.34,i+0.34],mode:'lines',line:{color:INK,width:2},hovertemplate:'unsteered: '+g.base+'%<extra></extra>'});
    ['gf','noise','fitv','sprobe'].forEach(k=>tr.push({x:[Math.max(0,g.base+g.sub[k])],y:[i+({sprobe:0,fitv:-0.2,gf:0,noise:0.22})[k]],mode:'markers',marker:NMK[k].m,hovertemplate:NMK[k].n+': %{x:.0f}%<extra></extra>'}));});
  Plotly.newPlot('n-sub',tr,L({height:270,margin:{l:112,r:16,t:10,b:46},xaxis:ax('hack rate (% of prompts)',{range:[-4,84]}),
    yaxis:ax('',{tickvals:[0,1],ticktext:rows.map(g=>g.name),range:[1.55,-0.55],gridcolor:'rgba(0,0,0,0)'})}),cfg);})();

// both signs: subtracted effect against added effect
(function(){const tr=[{x:[-60,60],y:[60,-60],mode:'lines',line:{color:'#cfc4b4',width:1,dash:'dot'},hoverinfo:'skip'},{x:[-60,60],y:[-60,60],mode:'lines',line:{color:'#cfc4b4',width:1},hoverinfo:'skip'}];
  tr.push({x:NT.rnd_sub.iso,y:NT.rnd_add.iso,mode:'markers',marker:ISO,hovertemplate:'random (isotropic)<br>subtracted %{x:+.0f}, added %{y:+.0f}<extra></extra>'});
  tr.push({x:NT.rnd_sub.sig,y:NT.rnd_add.sig,mode:'markers',marker:SIG,hovertemplate:'random (Σ-shaped)<br>subtracted %{x:+.0f}, added %{y:+.0f}<extra></extra>'});
  [['sprobe','top center'],['fitv','bottom center'],['gf','top right']].forEach(([k,pos])=>tr.push({x:[NT.sub[k]],y:[NT.add[k]],mode:'markers+text',marker:NMK[k].m,text:[NMK[k].n],textposition:pos,textfont:{size:12.5,color:'#5e5245'},hovertemplate:NMK[k].n+'<br>subtracted %{x:+.0f}, added %{y:+.0f}<extra></extra>'}));
  Plotly.newPlot('n-odd',tr,L({height:360,margin:{l:64,r:16,t:10,b:50},annotations:[{x:-50,y:50,text:'opposite effects<br>(a hack direction)',showarrow:false,font:{size:11.5,color:'#8a7d6d'}},{x:-48,y:-52,text:'lowers hacking<br>with either sign',showarrow:false,font:{size:11.5,color:'#8a7d6d'}}],
    xaxis:ax('change in hack rate when subtracted (points)',{range:[-62,42],zeroline:true,zerolinecolor:'#cfc4b4'}),yaxis:ax('change when added (points)',{range:[-62,62],zeroline:true,zerolinecolor:'#cfc4b4'})}),cfg);})();

// detection on rollouts: probe against Goodfire's direction, by family
(function(){const el=document.getElementById('n-roll'); if(!el||!NT.roll.probe) return; const d=NT.roll, E=Object.keys(d.probe);
  const NAME={ib_lcb:'ImpossibleBench<br>(mild wording)',answer_key:'answer key',leak_outputs:'leaked<br>outputs',release_gate:'release<br>gate',save_game:'save<br>game',scorer_copy:'scorer<br>copy'};
  const S=[['probe','logistic probe (single-turn fit)',{size:11,color:AI},-0.12,AI],['gf_pool','Goodfire direction',{size:11,symbol:'diamond',color:INK},0.12,INK]];
  const tr=S.map(([k,n,mk,dx,col])=>({x:E.map((_,i)=>i+dx),y:E.map(e=>d[k][e][0]),error_y:{type:'data',symmetric:false,array:E.map(e=>d[k][e][2]-d[k][e][0]),arrayminus:E.map(e=>d[k][e][0]-d[k][e][1]),color:col,thickness:1.2,width:0},mode:'markers',marker:mk,hovertemplate:n+': %{y:.2f}<extra></extra>'}));
  Plotly.newPlot('n-roll',tr,L({height:320,shapes:[{type:'line',x0:-0.5,x1:E.length-0.5,y0:0.5,y1:0.5,line:{color:'#cfc4b4',width:1,dash:'dot'}}],margin:{l:56,r:10,t:10,b:70},
    xaxis:ax('',{tickvals:E.map((_,i)=>i),ticktext:E.map(e=>(NAME[e]||e)+'<br>('+d.probe[e][3]+' tasks)'),gridcolor:'rgba(0,0,0,0)',range:[-0.5,E.length-0.5],tickfont:{size:11.5}}),yaxis:ax('matched AUROC',{range:[-0.03,1.04]})}),cfg);})();
