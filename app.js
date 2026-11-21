import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

const $ = (id) => document.getElementById(id);
const video = $("webcam");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");
const languageSelect = $("languageSelect");
const STORAGE_KEY = "signbridge-training-v1";

let handLandmarker = null;
let stream = null;
let cameraRunning = false;
let animationId = null;
let lastDetect = 0;
let latestFeatures = null;
let latestPrediction = null;
let training = false;
let trainingLabel = "";
let trainingTarget = 15;
let trainingCaptured = 0;
let trainingSamples = [];
let lastCaptureAt = 0;
let sentence = [];
let lastPredictedLabel = "";
let stableCount = 0;

function getStore() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{"ISL":[],"ASL":[]}'); }
  catch { return { ISL: [], ASL: [] }; }
}
function saveStore(data) { localStorage.setItem(STORAGE_KEY, JSON.stringify(data)); }
function lang() { return languageSelect.value; }
function toast(message) {
  const el = $("toast"); el.textContent = message; el.classList.add("show");
  clearTimeout(toast.timer); toast.timer = setTimeout(() => el.classList.remove("show"), 2600);
}
function setStatus(message) { $("systemStatus").textContent = message; }
function samplesForCurrentLanguage() { return getStore()[lang()] || []; }
function updateLibrary() {
  const data = samplesForCurrentLanguage();
  $("sampleCount").textContent = data.length;
  const groups = {};
  data.forEach(s => groups[s.label] = (groups[s.label] || 0) + 1);
  const list = $("libraryList");
  const labels = Object.keys(groups).sort();
  if (!labels.length) {
    list.innerHTML = '<div class="empty-library">No signs trained yet. Add your first sign to get started.</div>';
  } else {
    list.innerHTML = labels.map(label => `<div class="library-item"><div><strong>${escapeHtml(label)}</strong><span>${groups[label]} examples · ${lang()}</span></div><button class="delete-sign" data-label="${escapeHtml(label)}" title="Delete this sign">×</button></div>`).join("");
    list.querySelectorAll(".delete-sign").forEach(btn => btn.addEventListener("click", () => {
      const store = getStore(); store[lang()] = store[lang()].filter(s => s.label !== btn.dataset.label); saveStore(store);
      updateLibrary(); toast(`Deleted ${btn.dataset.label} examples.`);
    }));
  }
}
function escapeHtml(value) { return String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

// Convert 21 hand landmarks into a scale- and position-normalized feature vector.
function landmarksToFeatures(landmarks) {
  if (!landmarks || landmarks.length < 21) return null;
  const wrist = landmarks[0];
  const palmWidth = Math.hypot(landmarks[5].x - landmarks[17].x, landmarks[5].y - landmarks[17].y);
  const scale = palmWidth || 0.001;
  const values = [];
  for (const p of landmarks) {
    values.push((p.x - wrist.x) / scale, (p.y - wrist.y) / scale, (p.z - wrist.z) / scale);
  }
  return values;
}
function distance(a,b) {
  let sum = 0;
  for (let i=0;i<a.length;i++) { const d=a[i]-b[i]; sum += d*d; }
  return Math.sqrt(sum/a.length);
}
// A simple k-nearest-neighbour classifier trained from the user's own examples.
function predict(features) {
  const samples = samplesForCurrentLanguage();
  if (samples.length < 3) return null;
  const nearest = samples.map(s => ({label:s.label, d:distance(features,s.features)})).sort((a,b)=>a.d-b.d).slice(0,5);
  const votes = {};
  nearest.forEach((item, index) => {
    const weight = 1 / (item.d + 0.08);
    votes[item.label] = (votes[item.label] || 0) + weight;
  });
  const ranked = Object.entries(votes).sort((a,b)=>b[1]-a[1]);
  if (!ranked.length) return null;
  const total = ranked.reduce((sum,item)=>sum+item[1],0);
  const confidence = ranked[0][1]/total;
  const averageDistance = nearest.filter(x=>x.label===ranked[0][0]).reduce((s,x)=>s+x.d,0) / Math.max(1,nearest.filter(x=>x.label===ranked[0][0]).length);
  return {label:ranked[0][0], confidence:Math.min(confidence, Math.max(0, 1-averageDistance/2)), distance:averageDistance};
}
async function loadModel() {
  if (handLandmarker) return handLandmarker;
  setStatus("Loading hand tracking model…");
  const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", delegate: "GPU" },
    runningMode: "VIDEO", numHands: 1
  });
  setStatus("Hand tracking ready");
  return handLandmarker;
}
async function startCamera() {
  try {
    $("startCamera").disabled = true;
    await loadModel();
    stream = await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720},facingMode:"user"},audio:false});
    video.srcObject = stream;
    await video.play();
    cameraRunning = true;
    $("cameraPlaceholder").style.display = "none";
    $("cameraState").textContent = "LIVE"; $("cameraState").classList.add("online");
    $("trackingChip").classList.add("visible");
    $("stopCamera").disabled = false;
    $("recordSamples").disabled = false;
    $("startCamera").textContent = "Camera running";
    setStatus("Camera active · tracking");
    resizeCanvas();
    detectLoop();
  } catch (error) {
    console.error(error);
    $("startCamera").disabled = false;
    setStatus("Camera unavailable");
    toast(error.name === "NotAllowedError" ? "Camera permission was denied. Allow camera access in your browser." : "Could not start camera. Check your connection and camera.");
    $("cameraMessage").textContent = "Camera error: " + (error.message || error.name);
  }
}
function stopCamera() {
  cameraRunning = false;
  if (animationId) cancelAnimationFrame(animationId);
  if (stream) stream.getTracks().forEach(track => track.stop());
  stream = null; video.srcObject = null;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  $("cameraPlaceholder").style.display = "flex";
  $("cameraState").textContent = "OFFLINE"; $("cameraState").classList.remove("online");
  $("trackingChip").classList.remove("visible");
  $("stopCamera").disabled = true; $("startCamera").disabled = false; $("startCamera").innerHTML = 'Start camera <span>→</span>';
  $("recordSamples").disabled = true;
  setStatus("System ready");
  if (training) finishTraining("Camera stopped. Recording cancelled.");
}
function resizeCanvas() {
  const rect = video.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width * devicePixelRatio));
  canvas.height = Math.max(1, Math.round(rect.height * devicePixelRatio));
  ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
}
function drawLandmarks(landmarks) {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  ctx.clearRect(0,0,w,h);
  if (!landmarks) return;
  const connections = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
  ctx.lineWidth=2; ctx.strokeStyle="#6ce7e8"; ctx.shadowColor="#6ce7e8"; ctx.shadowBlur=7;
  connections.forEach(([a,b])=>{ctx.beginPath();ctx.moveTo(landmarks[a].x*w,landmarks[a].y*h);ctx.lineTo(landmarks[b].x*w,landmarks[b].y*h);ctx.stroke();});
  landmarks.forEach((p,i)=>{ctx.beginPath();ctx.arc(p.x*w,p.y*h,i%4===0?4:2.4,0,Math.PI*2);ctx.fillStyle=i%4===0?"#b4a0ff":"#eaffff";ctx.fill();});
  ctx.shadowBlur=0;
}
function detectLoop() {
  if (!cameraRunning) return;
  animationId = requestAnimationFrame(detectLoop);
  const now = performance.now();
  if (now - lastDetect < 70 || video.readyState < 2) return;
  lastDetect = now;
  try {
    const result = handLandmarker.detectForVideo(video, now);
    const landmarks = result.landmarks?.[0];
    drawLandmarks(landmarks);
    latestFeatures = landmarks ? landmarksToFeatures(landmarks) : null;
    if (!latestFeatures) {
      latestPrediction = null; stableCount=0; lastPredictedLabel="";
      $("prediction").textContent = "—"; $("confidence").textContent = "0%"; $("confidenceBar").style.width = "0%";
      $("predictionHint").textContent = "Show one hand clearly inside the camera frame.";
      $("addWord").disabled = true; $("speakPrediction").disabled = true;
      return;
    }
    if (training) captureTrainingSample(now);
    const resultPrediction = predict(latestFeatures);
    latestPrediction = resultPrediction;
    if (resultPrediction && resultPrediction.distance < 0.85 && resultPrediction.confidence >= 0.34) {
      if (lastPredictedLabel === resultPrediction.label) stableCount++; else {lastPredictedLabel=resultPrediction.label;stableCount=1;}
      const pct = Math.round(resultPrediction.confidence*100);
      $("prediction").textContent = resultPrediction.label;
      $("confidence").textContent = pct + "%"; $("confidenceBar").style.width = pct + "%";
      $("predictionHint").textContent = stableCount >= 5 ? "Sign detected. Add it to your sentence or speak it." : "Checking the sign… hold your hand steady.";
      $("addWord").disabled = stableCount < 5; $("speakPrediction").disabled = stableCount < 5;
    } else {
      stableCount=0; lastPredictedLabel="";
      $("prediction").textContent = "…"; $("confidence").textContent = "—"; $("confidenceBar").style.width = "0%";
      $("predictionHint").textContent = samplesForCurrentLanguage().length ? "No close match. Try a clearer pose or record more examples." : "Record examples in the Training Lab to teach the model your signs.";
      $("addWord").disabled = true; $("speakPrediction").disabled = true;
    }
  } catch (error) { console.warn("Frame processing error",error); }
}
function captureTrainingSample(now) {
  if (!latestFeatures || now-lastCaptureAt < 170) return;
  lastCaptureAt = now;
  trainingSamples.push({label:trainingLabel,features:latestFeatures});
  trainingCaptured++;
  const pct = Math.round(trainingCaptured/trainingTarget*100);
  $("recordProgressBar").style.width = pct+"%";
  $("recordStatus").textContent = `Recording ${trainingLabel}: ${trainingCaptured} / ${trainingTarget} samples`;
  if (trainingCaptured >= trainingTarget) finishTraining(`Saved ${trainingTarget} examples for ${trainingLabel}.`);
}
function beginTraining() {
  const label = $("signLabel").value.trim().toUpperCase().replace(/\s+/g," ");
  if (!label) {toast("Enter a sign label first.");$("signLabel").focus();return;}
  if (!cameraRunning) {toast("Start the camera before recording.");return;}
  if (training) return;
  training = true; trainingLabel = label; trainingCaptured = 0; trainingSamples = []; lastCaptureAt = 0;
  $("recordSamples").disabled = true;
  $("recordStatus").textContent = "Show the sign now. Keep your hand visible and move slightly between examples.";
  $("recordProgressBar").style.width = "0%";
  toast("Recording started. Hold your sign in view.");
}
function finishTraining(message) {
  if (trainingSamples.length) {
    const store = getStore();
    store[lang()] = [...(store[lang()]||[]), ...trainingSamples];
    saveStore(store); updateLibrary();
  }
  training = false; trainingSamples = [];
  $("recordSamples").disabled = !cameraRunning;
  $("recordStatus").textContent = message;
  if (message.startsWith("Saved")) toast(message);
}
function addToSentence(label) {
  if (!label) return;
  sentence.push(label);
  $("sentenceOutput").textContent = sentence.join(" ");
  $("sentenceOutput").classList.remove("empty");
  $("wordCount").textContent = `${sentence.length} ${sentence.length===1?"sign":"signs"}`;
}
function speak(text) {
  if (!("speechSynthesis" in window)) {toast("Speech output is not supported by this browser.");return;}
  if (!text.trim()) {toast("There is no text to speak yet.");return;}
  speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 0.88; utterance.pitch = 1;
  speechSynthesis.speak(utterance);
}
function resetPrediction() {
  $("prediction").textContent="—";$("confidence").textContent="0%";$("confidenceBar").style.width="0%";
  $("predictionHint").textContent="Record examples in the Training Lab to teach the model your signs.";
  $("addWord").disabled=true;$("speakPrediction").disabled=true;stableCount=0;lastPredictedLabel="";
}
$("startCamera").addEventListener("click",startCamera);
$("stopCamera").addEventListener("click",stopCamera);
$("mirrorToggle").addEventListener("change",e=>$("videoWrap").classList.toggle("no-mirror",!e.target.checked));
$("recordSamples").addEventListener("click",beginTraining);
$("addWord").addEventListener("click",()=>{if(latestPrediction)addToSentence(latestPrediction.label);});
$("speakPrediction").addEventListener("click",()=>{if(latestPrediction)speak(latestPrediction.label);});
$("speakSentence").addEventListener("click",()=>speak(sentence.join(" ")));
$("clearSentence").addEventListener("click",()=>{sentence=[];$("sentenceOutput").textContent="Your recognized signs will appear here…";$("sentenceOutput").classList.add("empty");$("wordCount").textContent="0 signs";});
$("languageSelect").addEventListener("change",()=>{if(training)finishTraining("Language changed. Recording cancelled.");updateLibrary();resetPrediction();});
$("deleteLanguageData").addEventListener("click",()=>{
  if(!confirm(`Delete all your ${lang()} training examples from this browser?`))return;
  const store=getStore();store[lang()]=[];saveStore(store);updateLibrary();resetPrediction();toast(`${lang()} library cleared.`);
});
window.addEventListener("resize",()=>{if(cameraRunning)resizeCanvas();});
updateLibrary();
$("sentenceOutput").classList.add("empty");
