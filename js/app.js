(function (global) {
  'use strict';

  const SONGS = global.SONGS;
  const midiToName = global.midiToName;
  const isBlackMidi = global.isBlackMidi;
  const formatTime = global.formatTime;
  const generateBeatGrid = global.generateBeatGrid;
  const parseStandardMidiFile = global.parseStandardMidiFile;
  const calculateAutoRange = global.calculateAutoRange;
  const assignHandsSmart = global.assignHandsSmart;
  const extendChordLegato = global.extendChordLegato;
  const audioEngine = global.audioEngine;
  const audioTranscriber = global.audioTranscriber;

  // Enrich built-in songs with metadata and save pristine factory copies for Reset
  const BUILTIN_IDS = ['let_it_be', 'imagine', 'clair_de_lune', 'gymnopedie', 'pyramid_song'];
  const FACTORY_BUILTIN_SONGS = {};
  BUILTIN_IDS.forEach(function (id) {
    const s = SONGS[id];
    if (!s) return;
    s.id = id;
    s.source = 'builtin';
    s.isCustom = false;
    s.isModifiedBuiltin = false;
    if (!s.bpm) {
      if (Array.isArray(s.beats) && s.beats.length >= 2) {
        const dt = s.beats[1].time - s.beats[0].time;
        s.bpm = dt > 0.05 ? Math.round(60 / dt) : 120;
      } else {
        s.bpm = 120;
      }
    }
    if (typeof s.gridOffset !== 'number') {
      s.gridOffset = (Array.isArray(s.beats) && s.beats.length > 0) ? s.beats[0].time : 0.8;
    }
    if (!s.beatsPerBar) {
      let bpb = 4;
      if (Array.isArray(s.beats)) {
        for (let i = 1; i < s.beats.length; i++) {
          if (s.beats[i].isMeasure) {
            bpb = i;
            break;
          }
        }
      }
      s.beatsPerBar = bpb;
    }
    FACTORY_BUILTIN_SONGS[id] = JSON.parse(JSON.stringify(s));
  });

  /********************************************************************
   * APPLICATION STATE
   ********************************************************************/
  function getDefaultHomeLayout() {
    try {
      const saved = localStorage.getItem('keyflow_home_layout');
      if (saved === 'cards' || saved === 'compact') {
        return saved;
      }
    } catch (e) {}
    const isTouchOrMobile =
      (window.matchMedia && window.matchMedia('(pointer: coarse)').matches) ||
      /iPad|iPhone|iPod|Android/i.test(navigator.userAgent || '') ||
      (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
    return isTouchOrMobile ? 'compact' : 'cards';
  }

  const state = {
    viewMode: 'home', // 'home' | 'play' | 'edit'
    homeFilter: 'all', // 'all' | 'ai' | 'midi' | 'builtin'
    homeSearch: '',
    homeLayout: getDefaultHomeLayout(), // 'cards' | 'compact'

    activeSongId: 'let_it_be',
    song: SONGS.let_it_be,
    isPlaying: false,
    currentTime: 0,
    speed: 1.0,
    horizonSec: 5.0, // Strict 5-second lookahead
    waitMode: false,
    waitingMidis: new Set(),
    waitingNoteIndices: new Set(),
    clearedWaitNoteIndices: new Set(),
    waitStartedAtMs: null,
    passHadLongWaitStall: false,
    passWrongKeys: 0,
    passMisses: 0,

    handIsoMaster: false,
    leftHandOn: false,
    rightHandOn: false,
    lastIsolatedHand: 'R',

    rangeMode: 'auto', // 'auto' | 'compact' | 'full88'
    showMeasureGrid: true,
    showAllKeyNames: false,
    hudMinimized: false,

    loopEnabled: false,
    loopA: 3.0,
    loopB: 14.0,
    speedRamp: false,
    loopPassCount: 1,
    targetRampSpeed: 1.0,

    // Phase 4A: Interactive Visual Note Editor State
    editorTool: 'select', // 'select' | 'group' | 'add' | 'delete'
    editorSnapBeats: 0.25, // 0 | 0.25 | 0.125 | 0.0625
    showSuspectNotes: true,
    selectedNoteIndex: -1,
    selectedNoteIndices: new Set(),
    groupDragBox: null,
    isCloningNote: false,
    undoStack: [],
    redoStack: [],

    // Phase 4B: Chord Detector, Metronome, Count-In & Accuracy
    showChordDetector: true,
    metronomeOn: false,
    countInOn: false,
    isCountingIn: false,
    songFinished: false,

    stats: { hits: 0, misses: 0, wrongKeys: 0, streak: 0, bestStreak: 0 },
    wrongKeyFlashes: {},
    hitScoredNoteIndices: new Set(),
    missedNoteIndices: new Set(),

    minMidi: 48,
    maxMidi: 84,
    whiteKeys: [],
    blackKeys: [],
    keyMap: {},

    playedIndices: new Set(),
    userPressed: {}, // midi -> { hand, time }
    particles: []
  };

  const pointerMidis = {};
  const heldKeys = {};

  function clearAllActiveInputs() {
    state.userPressed = {};
    state.wrongKeyFlashes = {};
    for (const k in heldKeys) delete heldKeys[k];
    for (const p in pointerMidis) delete pointerMidis[p];
  }

  window.addEventListener('blur', clearAllActiveInputs);
  document.addEventListener('visibilitychange', function () {
    if (document.hidden) clearAllActiveInputs();
  });

  function formatSpeedBadge(speed) {
    const rounded = Math.round(speed * 100) / 100;
    if (Math.abs(rounded - Math.round(rounded * 10) / 10) < 0.001) {
      return rounded.toFixed(1) + 'x';
    }
    return rounded.toFixed(2) + 'x';
  }

  /********************************************************************
   * FAULT-TOLERANT CANVAS SIZING & 60FPS RENDERER
   ********************************************************************/
  const stageEl = document.getElementById('stageContainer');
  const canvas = document.getElementById('pianoCanvas');
  const ctx = canvas.getContext('2d');

  let width = 960;
  let height = 500;
  let strikeY = 365;
  let keyboardH = 135;

  function syncViewportHeight() {
    const appShell = document.getElementById('appShell');
    if (!appShell) return;
    const isStandalone = window.navigator.standalone === true ||
      (window.matchMedia && window.matchMedia('(display-mode: standalone)').matches);
    if (isStandalone) {
      appShell.style.height = '100%';
      return;
    }
    const vh = (window.visualViewport && window.visualViewport.height > 100)
      ? window.visualViewport.height
      : (window.innerHeight || 0);
    if (vh > 100) {
      appShell.style.height = Math.round(vh) + 'px';
    }
  }

  function rebuildKeyboardGeometry() {
    const rect = stageEl ? stageEl.getBoundingClientRect() : { width: 960, height: 500 };
    width = (rect.width > 100) ? rect.width : (window.innerWidth || 960);
    height = (rect.height > 80) ? rect.height : Math.max(240, (window.innerHeight || 500) - 106);

    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    keyboardH = Math.round(Math.max(96, Math.min(165, height * 0.24)));
    strikeY = height - keyboardH;

    const whiteMidis = [];
    for (let m = state.minMidi; m <= state.maxMidi; m++) {
      if (!isBlackMidi(m)) whiteMidis.push(m);
    }

    const numWhite = Math.max(1, whiteMidis.length);
    const whiteW = width / numWhite;
    const blackW = whiteW * 0.62;
    const blackH = keyboardH * 0.63;

    state.whiteKeys = [];
    state.blackKeys = [];
    state.keyMap = {};

    for (let i = 0; i < whiteMidis.length; i++) {
      const m = whiteMidis[i];
      const x = i * whiteW;
      const kObj = {
        midi: m,
        name: midiToName(m),
        isBlack: false,
        x: x,
        y: strikeY,
        w: whiteW,
        h: keyboardH,
        laneX: x + whiteW * 0.08,
        laneW: whiteW * 0.84
      };
      state.whiteKeys.push(kObj);
      state.keyMap[m] = kObj;
    }

    for (let m = state.minMidi; m <= state.maxMidi; m++) {
      if (isBlackMidi(m)) {
        const prevWhite = state.keyMap[m - 1];
        if (!prevWhite) continue;
        const pc = m % 12;
        let shift = 0.5;
        if (pc === 1 || pc === 6) shift = 0.55;
        if (pc === 3 || pc === 10) shift = 0.45;

        const x = prevWhite.x + prevWhite.w - blackW * shift;
        const kObj = {
          midi: m,
          name: midiToName(m),
          isBlack: true,
          x: x,
          y: strikeY,
          w: blackW,
          h: blackH,
          laneX: x + blackW * 0.06,
          laneW: blackW * 0.88
        };
        state.blackKeys.push(kObj);
        state.keyMap[m] = kObj;
      }
    }
  }

  function drawRoundedRect(c, x, y, w, h, r) {
    const rad = Math.max(0, Math.min(r, w / 2, h / 2));
    c.beginPath();
    c.moveTo(x + rad, y);
    c.lineTo(x + w - rad, y);
    c.quadraticCurveTo(x + w, y, x + w, y + rad);
    c.lineTo(x + w, y + h - rad);
    c.quadraticCurveTo(x + w, y + h, x + w - rad, y + h);
    c.lineTo(x + rad, y + h);
    c.quadraticCurveTo(x, y + h, x, y + h - rad);
    c.lineTo(x, y + rad);
    c.quadraticCurveTo(x, y, x + rad, y);
    c.closePath();
  }

  function spawnParticles(x, w, hand) {
    if (state.particles.length > 75) return;
    let rgb = (hand === 'L') ? '56, 189, 248' : '74, 222, 128';
    if (hand === 'wrong') rgb = '239, 68, 68';
    const count = (hand === 'wrong') ? 4 : 2;
    for (let i = 0; i < count; i++) {
      state.particles.push({
        x: x + Math.random() * w,
        y: strikeY + (Math.random() * 4 - 2),
        vx: (Math.random() - 0.5) * (hand === 'wrong' ? 36 : 26),
        vy: -(16 + Math.random() * (hand === 'wrong' ? 46 : 38)),
        r: (hand === 'wrong' ? 1.6 : 1.4) + Math.random() * 2.0,
        alpha: 0.95,
        rgb: rgb
      });
    }
  }

  function renderParticles(dt) {
    for (let i = state.particles.length - 1; i >= 0; i--) {
      const p = state.particles[i];
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.alpha -= dt * 1.45;
      if (p.alpha <= 0) {
        state.particles.splice(i, 1);
        continue;
      }
      ctx.save();
      ctx.beginPath();
      ctx.fillStyle = 'rgba(' + p.rgb + ',' + p.alpha + ')';
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  }

  let editDrag = null;
  let lastEmptyRunwayTap = { timeMs: 0, x: 0, y: 0 };

  function getRunwayTiming() {
    const horizonReal = state.horizonSec;
    const speedFactor = (state.viewMode === 'edit') ? 1.0 : Math.max(0.25, state.speed || 1.0);
    const songHorizon = horizonReal * speedFactor;
    const pxPerSongSec = strikeY / songHorizon;
    const pxPerRealSec = strikeY / horizonReal;
    return { horizonReal, speedFactor, songHorizon, pxPerSongSec, pxPerRealSec };
  }

  function renderFrame(dt) {
    ctx.clearRect(0, 0, width, height);

    // 1. Deep Runway Background
    const bgGrad = ctx.createLinearGradient(0, 0, 0, strikeY);
    bgGrad.addColorStop(0, '#070b14');
    bgGrad.addColorStop(0.65, '#0b1220');
    bgGrad.addColorStop(1, '#101a2e');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, width, strikeY);

    // 2. Vertical Key Lane Guides
    for (let i = 0; i < state.whiteKeys.length; i++) {
      const wk = state.whiteKeys[i];
      const isC = (wk.midi % 12 === 0);
      ctx.beginPath();
      ctx.moveTo(wk.x, 0);
      ctx.lineTo(wk.x, strikeY);
      ctx.strokeStyle = isC ? 'rgba(148, 163, 184, 0.16)' : 'rgba(148, 163, 184, 0.045)';
      ctx.lineWidth = isC ? 1.2 : 1;
      ctx.stroke();
    }

    const { horizonReal, songHorizon, pxPerSongSec, pxPerRealSec } = getRunwayTiming();
    const curT = state.currentTime;

    // 3A. Scrolling Musical Measure & Beat Gridlines
    if (state.showMeasureGrid && Array.isArray(state.song.beats)) {
      const beats = state.song.beats;
      ctx.save();
      for (let i = 0; i < beats.length; i++) {
        const b = beats[i];
        if (b.time < curT) continue;
        if (b.time > curT + songHorizon + 0.05) break;

        const y = strikeY - (b.time - curT) * pxPerSongSec;
        if (y < 8 || y > strikeY - 2) continue;

        if (b.isMeasure) {
          ctx.beginPath();
          ctx.setLineDash([]);
          ctx.moveTo(0, y);
          ctx.lineTo(width - 42, y);
          ctx.strokeStyle = 'rgba(56, 189, 248, 0.22)';
          ctx.lineWidth = 1.5;
          ctx.stroke();

          // Measure number pill on the left side
          const labelText = 'BAR ' + b.barNumber;
          ctx.font = '700 9.5px ui-monospace, SFMono-Regular, Menlo, monospace';
          const textW = ctx.measureText(labelText).width;
          const pillW = textW + 10;
          const pillH = 15;
          const pillX = 10;
          const pillY = Math.max(4, y - pillH - 2);

          drawRoundedRect(ctx, pillX, pillY, pillW, pillH, 4);
          ctx.fillStyle = 'rgba(15, 23, 42, 0.78)';
          ctx.fill();
          ctx.strokeStyle = 'rgba(56, 189, 248, 0.35)';
          ctx.lineWidth = 1;
          ctx.stroke();

          ctx.fillStyle = 'rgba(125, 211, 252, 0.88)';
          ctx.textAlign = 'left';
          ctx.textBaseline = 'middle';
          ctx.fillText(labelText, pillX + 5, pillY + pillH / 2 + 0.5);
        } else {
          ctx.beginPath();
          ctx.setLineDash([4, 6]);
          ctx.moveTo(0, y);
          ctx.lineTo(width - 42, y);
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
      ctx.setLineDash([]);
      ctx.restore();
    }

    // 3A-Edit: Subtle Sub-Beat Snap Gridlines when in Edit Mode
    if (state.viewMode === 'edit' && state.editorSnapBeats > 0 && state.editorSnapBeats < 1) {
      const bpmVal = Math.max(30, Math.min(260, state.song.bpm || 120));
      const beatSec = 60 / bpmVal;
      const stepSec = beatSec * state.editorSnapBeats;
      const gridOff = typeof state.song.gridOffset === 'number' ? state.song.gridOffset : 0.8;
      if (stepSec >= 0.04) {
        const firstIdx = Math.ceil((curT - gridOff) / stepSec);
        const endT = curT + songHorizon + 0.05;
        ctx.save();
        ctx.beginPath();
        ctx.setLineDash([2, 6]);
        for (let k = firstIdx; k < firstIdx + 160; k++) {
          const tSnap = gridOff + k * stepSec;
          if (tSnap > endT) break;
          if (tSnap < curT) continue;
          const ySnap = strikeY - (tSnap - curT) * pxPerSongSec;
          if (ySnap < 8 || ySnap > strikeY - 2) continue;
          ctx.moveTo(0, ySnap);
          ctx.lineTo(width - 44, ySnap);
        }
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.065)';
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();
      }
    }

    // 3B. Horizontal 5-Second Horizon Markers (1s..5s on the right edge)
    ctx.save();
    ctx.font = '600 11px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    for (let s = 1; s <= Math.floor(horizonReal); s++) {
      const y = strikeY - s * pxPerRealSec;
      if (y < 14) continue;
      ctx.beginPath();
      ctx.moveTo(state.showMeasureGrid ? (width - 52) : 0, y);
      ctx.lineTo(width - 42, y);
      ctx.strokeStyle = state.showMeasureGrid ? 'rgba(148, 163, 184, 0.28)' : 'rgba(255, 255, 255, 0.075)';
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = 'rgba(148, 163, 184, 0.58)';
      ctx.fillText(s + 's', width - 12, y);
    }
    ctx.restore();

    // 4. Falling Note Bars & Explicit Note Badges
    const activeKeys = {};
    Object.keys(state.userPressed).forEach(mStr => {
      const m = parseInt(mStr, 10);
      activeKeys[m] = { hand: state.userPressed[m].hand };
    });

    const notes = state.song.notes;

    for (let i = 0; i < notes.length; i++) {
      const n = notes[i];
      const nStart = n.time;
      const nEnd = n.time + n.duration;

      if (nEnd < curT - 0.04) continue;
      if (nStart > curT + songHorizon + 0.1) {
        if (state.viewMode === 'edit' && editDrag && editDrag.noteRef) continue;
        break;
      }

      const kGeom = state.keyMap[n.midi];
      if (!kGeom) continue;

      const handActive = !state.handIsoMaster ||
        (n.hand === 'L' && state.leftHandOn) ||
        (n.hand === 'R' && state.rightHandOn);

      const rawBotY = strikeY - (nStart - curT) * pxPerSongSec;
      const rawTopY = strikeY - (nEnd - curT) * pxPerSongSec;

      const botY = Math.min(strikeY, rawBotY);
      const topY = Math.max(-16, rawTopY);
      const barH = botY - topY;
      if (barH <= 2) continue;

      const hittingNow = (nStart <= curT + 0.025 && nEnd >= curT);
      const isWaiting = state.waitingNoteIndices.has(i);

      if (hittingNow && handActive) {
        activeKeys[n.midi] = { hand: n.hand };
        if (state.isPlaying && state.waitingMidis.size === 0 && Math.random() < 0.3) {
          spawnParticles(kGeom.laneX, kGeom.laneW, n.hand);
        }
      }

      // Subtle vertical lane highlight under approaching notes
      if (handActive && rawBotY > strikeY * 0.25) {
        const alpha = Math.min(0.11, Math.max(0, 1 - (nStart - curT) / songHorizon) * 0.11);
        ctx.fillStyle = (n.hand === 'L')
          ? 'rgba(41, 151, 255, ' + alpha + ')'
          : 'rgba(52, 211, 153, ' + alpha + ')';
        ctx.fillRect(kGeom.laneX, botY, kGeom.laneW, strikeY - botY);
      }

      ctx.save();
      ctx.globalAlpha = handActive ? 1.0 : 0.16;

      const x = kGeom.laneX;
      const w = kGeom.laneW;
      const rad = Math.min(7, w * 0.28);

      const barGrad = ctx.createLinearGradient(x, topY, x, botY);
      if (n.hand === 'L') {
        barGrad.addColorStop(0, '#7dd3fc');
        barGrad.addColorStop(0.3, '#2997ff');
        barGrad.addColorStop(1, '#0284c7');
        ctx.shadowColor = hittingNow ? 'rgba(56, 189, 248, 0.95)' : 'rgba(41, 151, 255, 0.42)';
      } else {
        barGrad.addColorStop(0, '#86efac');
        barGrad.addColorStop(0.3, '#34d399');
        barGrad.addColorStop(1, '#16a34a');
        ctx.shadowColor = hittingNow ? 'rgba(74, 222, 128, 0.95)' : 'rgba(52, 211, 153, 0.42)';
      }
      ctx.shadowBlur = hittingNow ? 16 : 9;

      drawRoundedRect(ctx, x, topY, w, barH, rad);
      ctx.fillStyle = barGrad;
      ctx.fill();

      const isInGroup = (state.viewMode === 'edit' && state.selectedNoteIndices && state.selectedNoteIndices.has(i));
      const isSelectedInEdit = (state.viewMode === 'edit' && (i === state.selectedNoteIndex || isInGroup));
      if (isInGroup) {
        ctx.shadowColor = '#38bdf8';
        ctx.shadowBlur = 12;
      } else {
        ctx.shadowBlur = 0;
      }
      ctx.lineWidth = isSelectedInEdit ? 2.5 : (isWaiting ? 2.2 : 1.2);
      ctx.strokeStyle = isSelectedInEdit
        ? '#ffffff'
        : (isWaiting
          ? '#ffffff'
          : (n.hand === 'L' ? 'rgba(186, 230, 253, 0.85)' : 'rgba(187, 247, 208, 0.85)'));
      ctx.stroke();
      ctx.shadowBlur = 0;

      if (state.viewMode === 'edit' && state.showSuspectNotes && n.suspect) {
        ctx.save();
        drawRoundedRect(ctx, x - 1, topY - 1, w + 2, barH + 2, rad + 1);
        ctx.strokeStyle = isSelectedInEdit ? '#ffffff' : '#f59e0b';
        ctx.lineWidth = isSelectedInEdit ? 2.8 : 2.2;
        ctx.setLineDash([5, 3]);
        ctx.stroke();
        ctx.setLineDash([]);

        if (w >= 12 && barH >= 14) {
          const warnR = Math.min(7, Math.max(5, w * 0.22));
          const warnX = x + warnR + 2;
          const warnY = Math.min(botY - warnR - 2, topY + warnR + 3);
          ctx.beginPath();
          ctx.arc(warnX, warnY, warnR, 0, Math.PI * 2);
          ctx.fillStyle = '#f59e0b';
          ctx.fill();
          ctx.lineWidth = 1.1;
          ctx.strokeStyle = '#0f172a';
          ctx.stroke();
          ctx.font = '800 ' + Math.round(warnR * 1.4) + 'px -apple-system, BlinkMacSystemFont, sans-serif';
          ctx.fillStyle = '#0f172a';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText('!', warnX, warnY + 0.5);
        }
        ctx.restore();
      }

      // Group Select Checkmark Badge when multiple notes are selected or in Group Select tool mode
      if (isInGroup && (state.selectedNoteIndices.size >= 2 || state.editorTool === 'group') && w >= 12 && barH >= 14) {
        const chkR = Math.min(7, Math.max(5, w * 0.22));
        const chkX = x + w - chkR - 2;
        const chkY = Math.min(botY - chkR - 2, topY + chkR + 3);
        ctx.beginPath();
        ctx.arc(chkX, chkY, chkR, 0, Math.PI * 2);
        ctx.fillStyle = '#38bdf8';
        ctx.fill();
        ctx.lineWidth = 1.2;
        ctx.strokeStyle = '#ffffff';
        ctx.stroke();
        ctx.font = '700 ' + Math.round(chkR * 1.35) + 'px -apple-system, BlinkMacSystemFont, sans-serif';
        ctx.fillStyle = '#0f172a';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('✓', chkX, chkY + 0.5);
      }

      // Top Resize Handle Bar when in Edit Mode (bright on selected note, subtle on others)
      if (state.viewMode === 'edit' && state.editorTool !== 'group' && w >= 12 && barH >= 14) {
        const handleH = isSelectedInEdit ? 8 : 5;
        const handleW = Math.max(6, w - 4);
        const handleX = x + 2;
        const handleY = topY + 1;
        drawRoundedRect(ctx, handleX, handleY, handleW, handleH, 3);
        ctx.fillStyle = isSelectedInEdit ? '#ffffff' : 'rgba(255, 255, 255, 0.35)';
        ctx.fill();
        if (isSelectedInEdit && handleW >= 12) {
          ctx.beginPath();
          ctx.moveTo(handleX + 4, handleY + handleH / 2);
          ctx.lineTo(handleX + handleW - 4, handleY + handleH / 2);
          ctx.strokeStyle = '#0f172a';
          ctx.lineWidth = 1.3;
          ctx.stroke();
        }
      }

      // Explicit Note Name Badge & Finger Number Badge inside every falling bar
      const badgeH = Math.min(20, Math.max(14, w * 0.52));
      if (barH >= badgeH + 2 && w >= 14) {
        const badgeW = Math.max(w - 4, 14);
        const badgeX = x + (w - badgeW) / 2;
        const badgeY = Math.max(topY + 2, botY - badgeH - 4);

        drawRoundedRect(ctx, badgeX, badgeY, badgeW, badgeH, 4);
        ctx.fillStyle = 'rgba(15, 23, 42, 0.42)';
        ctx.fill();

        const fSize = Math.min(12, Math.max(8.5, w * 0.36));
        ctx.font = '700 ' + fSize + 'px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const labelText = (n.finger >= 1 && n.finger <= 5 && w >= 28)
          ? (n.name + ' • ' + n.finger)
          : n.name;
        ctx.fillText(labelText, x + w / 2, badgeY + badgeH / 2 + 0.5);

        // Circular finger badge near trailing edge if finger is assigned (1-5)
        if (n.finger >= 1 && n.finger <= 5 && barH >= badgeH + 18) {
          const fR = Math.min(8, Math.max(5.5, w * 0.3));
          const fX = x + w / 2;
          const fY = Math.min(botY - badgeH - fR - 5, topY + fR + 4);
          if (fY > topY + fR) {
            ctx.beginPath();
            ctx.arc(fX, fY, fR, 0, Math.PI * 2);
            ctx.fillStyle = '#f59e0b';
            ctx.fill();
            ctx.lineWidth = 1.2;
            ctx.strokeStyle = '#ffffff';
            ctx.stroke();
            ctx.font = '700 ' + Math.round(fR * 1.3) + 'px -apple-system, BlinkMacSystemFont, sans-serif';
            ctx.fillStyle = '#0f172a';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(String(n.finger), fX, fY + 0.5);
          }
        }
      }

      ctx.restore();
    }

    // 4B. Group Select Marquee Box on Runway in Edit Mode
    if (state.viewMode === 'edit' && state.groupDragBox) {
      const bx = Math.min(state.groupDragBox.x0, state.groupDragBox.x1);
      const by = Math.min(state.groupDragBox.y0, state.groupDragBox.y1);
      const bw = Math.abs(state.groupDragBox.x1 - state.groupDragBox.x0);
      const bh = Math.abs(state.groupDragBox.y1 - state.groupDragBox.y0);
      if (bw > 2 || bh > 2) {
        ctx.save();
        ctx.fillStyle = 'rgba(56, 189, 248, 0.14)';
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.85)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 4]);
        ctx.fillRect(bx, by, bw, bh);
        ctx.strokeRect(bx, by, bw, bh);
        ctx.restore();
      }
    }

    // 5. Glowing "NOW" Strike Line & Contact Flares
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(0, strikeY);
    ctx.lineTo(width - 44, strikeY);
    ctx.strokeStyle = 'rgba(226, 232, 240, 0.8)';
    ctx.lineWidth = 2;
    ctx.shadowColor = 'rgba(56, 189, 248, 0.8)';
    ctx.shadowBlur = 10;
    ctx.stroke();

    ctx.shadowBlur = 0;
    ctx.font = '700 11px -apple-system, BlinkMacSystemFont, sans-serif';
    ctx.fillStyle = '#f8fafc';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText('NOW', width - 8, strikeY - 1);
    ctx.restore();

    Object.keys(activeKeys).forEach(mStr => {
      const m = parseInt(mStr, 10);
      const kGeom = state.keyMap[m];
      if (!kGeom) return;
      const info = activeKeys[m];
      const cx = kGeom.laneX + kGeom.laneW / 2;
      const r = kGeom.laneW * 0.95;

      const flare = ctx.createRadialGradient(cx, strikeY, 1, cx, strikeY, r);
      if (info.hand === 'L') {
        flare.addColorStop(0, 'rgba(255, 255, 255, 0.96)');
        flare.addColorStop(0.35, 'rgba(56, 189, 248, 0.82)');
        flare.addColorStop(1, 'rgba(56, 189, 248, 0)');
      } else {
        flare.addColorStop(0, 'rgba(255, 255, 255, 0.96)');
        flare.addColorStop(0.35, 'rgba(74, 222, 128, 0.82)');
        flare.addColorStop(1, 'rgba(74, 222, 128, 0)');
      }
      ctx.save();
      ctx.fillStyle = flare;
      ctx.beginPath();
      ctx.arc(cx, strikeY, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    });

    // Wrong Note Contact Flares (Red)
    const nowMs = performance.now();
    Object.keys(state.wrongKeyFlashes).forEach(mStr => {
      const m = parseInt(mStr, 10);
      const elapsed = nowMs - state.wrongKeyFlashes[m];
      if (elapsed > 350) {
        delete state.wrongKeyFlashes[m];
        return;
      }
      const kGeom = state.keyMap[m];
      if (!kGeom) return;
      const cx = kGeom.laneX + kGeom.laneW / 2;
      const r = kGeom.laneW * 1.1;
      const alpha = Math.max(0, 1 - elapsed / 350);

      const flare = ctx.createRadialGradient(cx, strikeY, 1, cx, strikeY, r);
      flare.addColorStop(0, 'rgba(255, 255, 255, ' + (0.95 * alpha) + ')');
      flare.addColorStop(0.35, 'rgba(239, 68, 68, ' + (0.85 * alpha) + ')');
      flare.addColorStop(1, 'rgba(239, 68, 68, 0)');
      ctx.save();
      ctx.fillStyle = flare;
      ctx.beginPath();
      ctx.arc(cx, strikeY, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    });

    renderParticles(dt);

    // 6. Auto-Scaled 3D Virtual Piano Keyboard
    // White Keys
    for (let i = 0; i < state.whiteKeys.length; i++) {
      const wk = state.whiteKeys[i];
      const act = activeKeys[wk.midi];
      const isWrong = (state.wrongKeyFlashes[wk.midi] && (nowMs - state.wrongKeyFlashes[wk.midi] < 350));
      const kx = wk.x + 0.8;
      const ky = wk.y;
      const kw = wk.w - 1.6;
      const kh = wk.h - 4;

      ctx.save();
      drawRoundedRect(ctx, kx, ky, kw, kh, 5);
      const kGrad = ctx.createLinearGradient(kx, ky, kx, ky + kh);
      if (isWrong) {
        kGrad.addColorStop(0, '#fecaca');
        kGrad.addColorStop(0.25, '#ef4444');
        kGrad.addColorStop(1, '#b91c1c');
      } else if (act) {
        if (act.hand === 'L') {
          kGrad.addColorStop(0, '#bae6fd');
          kGrad.addColorStop(0.25, '#38bdf8');
          kGrad.addColorStop(1, '#0284c7');
        } else {
          kGrad.addColorStop(0, '#bbf7d0');
          kGrad.addColorStop(0.25, '#4ade80');
          kGrad.addColorStop(1, '#16a34a');
        }
      } else {
        kGrad.addColorStop(0, '#e2e8f0');
        kGrad.addColorStop(0.12, '#f8fafc');
        kGrad.addColorStop(0.92, '#ffffff');
        kGrad.addColorStop(1, '#cbd5e1');
      }
      ctx.fillStyle = kGrad;
      ctx.fill();

      ctx.fillStyle = isWrong ? '#991b1b' : (act ? (act.hand === 'L' ? '#0369a1' : '#15803d') : '#94a3b8');
      ctx.fillRect(kx + 1, ky + kh - 5, kw - 2, 5);

      ctx.strokeStyle = isWrong ? '#ef4444' : '#0f172a';
      ctx.lineWidth = isWrong ? 2 : 1;
      ctx.stroke();

      const isC = (wk.midi % 12 === 0);
      if ((isC || state.showAllKeyNames || act || isWrong) && kw >= 15) {
        const fSize = Math.min(12.5, Math.max(8.5, kw * 0.34));
        ctx.font = (isC || act || isWrong ? '700 ' : '600 ') + fSize + 'px -apple-system, BlinkMacSystemFont, sans-serif';
        ctx.fillStyle = isWrong ? '#ffffff' : (act ? '#082f49' : (isC ? '#0f172a' : '#64748b'));
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText(wk.name, kx + kw / 2, ky + kh - 8);
      }
      ctx.restore();
    }

    // Black Keys
    for (let i = 0; i < state.blackKeys.length; i++) {
      const bk = state.blackKeys[i];
      const act = activeKeys[bk.midi];
      const isWrong = (state.wrongKeyFlashes[bk.midi] && (nowMs - state.wrongKeyFlashes[bk.midi] < 350));
      const kx = bk.x;
      const ky = bk.y;
      const kw = bk.w;
      const kh = bk.h;

      ctx.save();
      drawRoundedRect(ctx, kx, ky, kw, kh, 4);
      const bGrad = ctx.createLinearGradient(kx, ky, kx, ky + kh);
      if (isWrong) {
        bGrad.addColorStop(0, '#f87171');
        bGrad.addColorStop(0.35, '#dc2626');
        bGrad.addColorStop(1, '#7f1d1d');
      } else if (act) {
        if (act.hand === 'L') {
          bGrad.addColorStop(0, '#7dd3fc');
          bGrad.addColorStop(0.4, '#0284c7');
          bGrad.addColorStop(1, '#0c4a6e');
        } else {
          bGrad.addColorStop(0, '#86efac');
          bGrad.addColorStop(0.4, '#16a34a');
          bGrad.addColorStop(1, '#14532d');
        }
      } else {
        bGrad.addColorStop(0, '#334155');
        bGrad.addColorStop(0.15, '#1e293b');
        bGrad.addColorStop(1, '#090d16');
      }
      ctx.fillStyle = bGrad;
      ctx.fill();

      if (kw >= 10) {
        drawRoundedRect(ctx, kx + 2, ky + 1, kw - 4, kh - 7, 2.5);
        ctx.strokeStyle = isWrong ? 'rgba(255, 255, 255, 0.85)' : (act ? 'rgba(255, 255, 255, 0.55)' : 'rgba(255, 255, 255, 0.13)');
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      if ((state.showAllKeyNames || act || isWrong) && kw >= 16) {
        const fSize = Math.min(10, Math.max(7.5, kw * 0.36));
        ctx.font = '600 ' + fSize + 'px -apple-system, BlinkMacSystemFont, sans-serif';
        ctx.fillStyle = (act || isWrong) ? '#ffffff' : 'rgba(226, 232, 240, 0.85)';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText(bk.name, kx + kw / 2, ky + kh - 7);
      }
      ctx.restore();
    }
  }

  /********************************************************************
   * PLAYBACK ENGINE, WAIT MODE & A-B LOOPING
   ********************************************************************/
  let lastTimeMs = performance.now();

  function updateWaitBanner() {
    const banner = document.getElementById('waitBanner');
    const txt = document.getElementById('waitBannerText');
    if (state.viewMode !== 'edit' && state.waitMode && state.waitingMidis.size > 0) {
      banner.classList.add('visible');
      const list = Array.from(state.waitingMidis).sort((a, b) => a - b).map(midiToName);
      txt.textContent = 'Wait Mode: Play ' + list.join(' + ');
    } else {
      banner.classList.remove('visible');
    }
  }

  function resetNoteTriggersBefore(sec) {
    state.playedIndices.clear();
    state.clearedWaitNoteIndices.clear();
    state.waitingNoteIndices.clear();
    state.waitingMidis.clear();
    state.missedNoteIndices.clear();
    state.waitStartedAtMs = null;
    updateWaitBanner();

    const notes = state.song.notes;
    for (let i = 0; i < notes.length; i++) {
      if (notes[i].time < sec - 0.02) {
        state.playedIndices.add(i);
        state.clearedWaitNoteIndices.add(i);
        state.missedNoteIndices.add(i);
      }
    }
  }

  function advanceClock(dt) {
    if (!state.isPlaying) return;
    if (state.waitMode && state.waitingMidis.size > 0) {
      if (state.waitStartedAtMs !== null && (performance.now() - state.waitStartedAtMs) > 3000) {
        state.passHadLongWaitStall = true;
      }
      return;
    }

    const prevT = state.currentTime;
    let nextT = prevT + dt * state.speed;
    const notes = state.song.notes;

    if (state.waitMode) {
      for (let i = 0; i < notes.length; i++) {
        const n = notes[i];
        if (n.time > nextT) break;
        const handActive = !state.handIsoMaster ||
          (n.hand === 'L' && state.leftHandOn) ||
          (n.hand === 'R' && state.rightHandOn);
        if (!handActive) continue;

        if (!state.clearedWaitNoteIndices.has(i) && n.time >= prevT - 0.02 && n.time <= nextT) {
          nextT = Math.max(prevT, n.time);
          for (let j = i; j < notes.length; j++) {
            const nj = notes[j];
            if (nj.time - n.time > 0.035) break;
            if (state.clearedWaitNoteIndices.has(j)) continue;
            const hAct = !state.handIsoMaster ||
              (nj.hand === 'L' && state.leftHandOn) ||
              (nj.hand === 'R' && state.rightHandOn);
            if (hAct) {
              state.waitingNoteIndices.add(j);
              state.waitingMidis.add(nj.midi);
            }
          }
          state.waitStartedAtMs = performance.now();
          updateWaitBanner();
          break;
        }
      }
    }

    // Metronome Click on Musical Beat Crossing
    if (state.metronomeOn) {
      if (Array.isArray(state.song.beats) && state.song.beats.length > 0) {
        for (let bIdx = 0; bIdx < state.song.beats.length; bIdx++) {
          const b = state.song.beats[bIdx];
          if (b.time >= prevT && b.time < nextT) {
            audioEngine.playMetronomeClick(b.isMeasure);
          } else if (b.time >= nextT) {
            break;
          }
        }
      } else {
        const bpmVal = state.song.bpm || 120;
        const bSec = (60 / bpmVal);
        const bpb = state.song.beatsPerBar || 4;
        const off = typeof state.song.gridOffset === 'number' ? state.song.gridOffset : 0;
        const prevBeat = Math.floor((prevT - off) / bSec);
        const nextBeat = Math.floor((nextT - off) / bSec);
        if (nextBeat > prevBeat) {
          const isMeasure = ((nextBeat % bpb) + bpb) % bpb === 0;
          audioEngine.playMetronomeClick(isMeasure);
        }
      }
    }

    state.currentTime = nextT;

    for (let i = 0; i < notes.length; i++) {
      if (state.playedIndices.has(i)) continue;
      const n = notes[i];
      if (n.time <= state.currentTime) {
        if (state.waitMode && state.waitingNoteIndices.has(i)) continue;
        state.playedIndices.add(i);
        const handActive = !state.handIsoMaster ||
          (n.hand === 'L' && state.leftHandOn) ||
          (n.hand === 'R' && state.rightHandOn);
        if (handActive) {
          audioEngine.playNote(n.midi, n.duration / state.speed, n.velocity);
        }
      } else {
        break;
      }
    }

    // Count missed notes once they pass the 300ms late-hit tolerance window (if user is actively playing)
    if (state.viewMode === 'play' && (state.stats.hits + state.stats.wrongKeys) > 0) {
      const missCutoff = state.currentTime - 0.30;
      let missedAny = false;
      for (let i = 0; i < notes.length; i++) {
        const n = notes[i];
        if (n.time >= missCutoff) break;
        if (state.hitScoredNoteIndices.has(i) || state.missedNoteIndices.has(i)) continue;
        const handActive = !state.handIsoMaster ||
          (n.hand === 'L' && state.leftHandOn) ||
          (n.hand === 'R' && state.rightHandOn);
        state.missedNoteIndices.add(i);
        if (handActive) {
          state.stats.misses++;
          state.passMisses++;
          state.stats.streak = 0;
          missedAny = true;
        }
      }
      if (missedAny) updateAccuracyUI();
    }

    if (state.loopEnabled && state.currentTime >= state.loopB) {
      state.loopPassCount++;
      const isCleanPass = !state.passHadLongWaitStall && state.passWrongKeys === 0 && state.passMisses === 0;
      if (state.speedRamp && isCleanPass && state.speed < state.targetRampSpeed) {
        applyPlaybackSpeed(Math.min(state.targetRampSpeed, Math.round((state.speed + 0.05) * 100) / 100));
      }
      state.passHadLongWaitStall = false;
      state.passWrongKeys = 0;
      state.passMisses = 0;
      state.waitStartedAtMs = null;
      syncLoopUI();
      seekTo(state.loopA);
      if (state.countInOn) {
        doSetPlaying(false);
        runCountIn(function () {
          doSetPlaying(true);
        });
      }
      return;
    }

    if (state.currentTime >= state.song.duration) {
      const totalFinal = state.stats.hits + state.stats.misses + state.stats.wrongKeys;
      if (totalFinal >= 8 && state.stats.hits >= 4 && state.song && state.song.id) {
        recordSongBestAccuracy(state.song.id, Math.round((state.stats.hits / totalFinal) * 100));
      }
      state.currentTime = 0;
      resetNoteTriggersBefore(0);
      setPlaying(false);
      state.songFinished = true;
    }

    syncTimelineUI();
  }

  function seekTo(sec) {
    if (audioEngine && typeof audioEngine.stopAll === 'function') {
      audioEngine.stopAll();
    }
    state.currentTime = Math.max(0, Math.min(state.song.duration, sec));
    resetNoteTriggersBefore(state.currentTime);
    resetAccuracyStats();
    state.passWrongKeys = 0;
    state.passMisses = 0;
    state.passHadLongWaitStall = false;
    cancelCountIn();
    syncTimelineUI();
  }

  let countInTimeouts = [];

  function cancelCountIn() {
    state.isCountingIn = false;
    countInTimeouts.forEach(function (t) { clearTimeout(t); });
    countInTimeouts = [];
    const overlay = document.getElementById('countInOverlay');
    if (overlay) overlay.style.display = 'none';
  }

  function runCountIn(onComplete) {
    cancelCountIn();
    state.isCountingIn = true;
    const overlay = document.getElementById('countInOverlay');
    if (overlay) overlay.style.display = 'flex';

    const bpmVal = Math.max(30, Math.min(260, state.song.bpm || 120));
    const beatSec = (60 / bpmVal) / Math.max(0.2, state.speed);
    const beatsCount = state.song.beatsPerBar || 4;

    for (let b = 1; b <= beatsCount; b++) {
      const delayMs = (b - 1) * (beatSec * 1000);
      const isDownbeat = (b === 1);
      const tid = setTimeout(function () {
        if (!state.isCountingIn) return;
        if (overlay) {
          overlay.textContent = String(b);
          overlay.classList.remove('beat');
          void overlay.offsetWidth;
          overlay.classList.add('beat');
        }
        audioEngine.playMetronomeClick(isDownbeat);
      }, delayMs);
      countInTimeouts.push(tid);
    }

    const endTid = setTimeout(function () {
      if (!state.isCountingIn) return;
      cancelCountIn();
      if (typeof onComplete === 'function') onComplete();
    }, beatsCount * (beatSec * 1000));
    countInTimeouts.push(endTid);
  }

  function setPlaying(play) {
    if (play && state.isCountingIn) {
      cancelCountIn();
      doSetPlaying(false);
      return;
    }
    if (play && !state.hudMinimized && typeof setHudMinimized === 'function') {
      setHudMinimized(true);
    }
    if (play && state.songFinished) {
      resetAccuracyStats();
    }
    if (play && state.countInOn && !state.isCountingIn) {
      runCountIn(function () {
        doSetPlaying(true);
      });
      return;
    }
    cancelCountIn();
    doSetPlaying(play);
  }

  function doSetPlaying(play) {
    const wasPlaying = state.isPlaying;
    state.isPlaying = play;
    if (play) {
      audioEngine.ensureContext();
    } else if (wasPlaying && audioEngine && typeof audioEngine.stopAll === 'function') {
      audioEngine.stopAll();
    }
    document.getElementById('playSvg').style.display = play ? 'none' : 'block';
    document.getElementById('pauseSvg').style.display = play ? 'block' : 'none';
  }

  function syncTimelineUI() {
    const dur = state.song.duration || 1;
    const pct = Math.max(0, Math.min(100, (state.currentTime / dur) * 100));
    document.getElementById('timelineFill').style.width = pct + '%';
    document.getElementById('playheadThumb').style.left = pct + '%';
    document.getElementById('topProgressFill').style.width = pct + '%';
    document.getElementById('curTimeText').textContent = formatTime(state.currentTime);
    document.getElementById('totTimeText').textContent = formatTime(dur);
  }

  function syncLoopUI() {
    const dur = state.song.duration || 1;
    const aPct = Math.max(0, Math.min(94, (state.loopA / dur) * 100));
    const bPct = Math.max(aPct + 4, Math.min(100, (state.loopB / dur) * 100));
    const box = document.getElementById('abLoopBox');
    const hA = document.getElementById('loopHandleA');
    const hB = document.getElementById('loopHandleB');

    box.style.left = aPct + '%';
    box.style.width = (bPct - aPct) + '%';
    hA.style.left = aPct + '%';
    hB.style.left = bPct + '%';

    if (state.loopEnabled && state.speedRamp) {
      box.textContent = 'LOOP • PASS ' + state.loopPassCount + ' • ' + state.speed.toFixed(2) + 'x';
    } else if (state.loopEnabled && state.loopPassCount > 1) {
      box.textContent = 'A–B LOOP • PASS ' + state.loopPassCount;
    } else {
      box.textContent = 'A–B LOOP';
    }
  }

  function tickLoop(now) {
    const dt = Math.min(0.05, Math.max(0.001, (now - lastTimeMs) / 1000));
    lastTimeMs = now;

    if (state.viewMode === 'home') {
      requestAnimationFrame(tickLoop);
      return;
    }

    advanceClock(dt);
    renderFrame(dt);
    updateLiveChordUI();

    requestAnimationFrame(tickLoop);
  }

  /********************************************************************
   * PHASE 4B: LIVE ACCURACY TRACKING & WRONG NOTE FLASHES
   ********************************************************************/
  const BEST_ACCURACY_STORAGE_KEY = 'keyflow_best_accuracy_v1';

  function loadBestAccuracyMap() {
    try {
      const raw = window.localStorage && window.localStorage.getItem(BEST_ACCURACY_STORAGE_KEY);
      return raw ? JSON.parse(raw) : {};
    } catch (_) {
      return {};
    }
  }

  function recordSongBestAccuracy(songId, pct) {
    if (!songId || !Number.isFinite(pct)) return;
    try {
      const map = loadBestAccuracyMap();
      if (typeof map[songId] !== 'number' || pct > map[songId]) {
        map[songId] = pct;
        if (window.localStorage) {
          window.localStorage.setItem(BEST_ACCURACY_STORAGE_KEY, JSON.stringify(map));
        }
      }
    } catch (_) {}
  }

  function updateAccuracyUI() {
    const pill = document.getElementById('accuracyHudPill');
    const pctEl = document.getElementById('accuracyPctText');
    const streakEl = document.getElementById('streakCountText');
    if (!pill || !pctEl || !streakEl) return;
    const total = state.stats.hits + state.stats.misses + state.stats.wrongKeys;
    if (total === 0) {
      pctEl.textContent = '100%';
      streakEl.textContent = '0 streak';
      pill.style.display = 'none';
      return;
    }
    pill.style.display = 'inline-flex';
    const pct = Math.round((state.stats.hits / total) * 100);
    pctEl.textContent = pct + '%';
    streakEl.textContent = state.stats.streak + ' streak';
  }

  function resetAccuracyStats() {
    state.songFinished = false;
    state.stats.hits = 0;
    state.stats.wrongKeys = 0;
    state.stats.misses = 0;
    state.stats.streak = 0;
    state.stats.bestStreak = 0;
    state.hitScoredNoteIndices.clear();
    state.missedNoteIndices.clear();
    updateAccuracyUI();
  }

  /********************************************************************
   * PHASE 4B: LIVE CHORD SYMBOL DETECTOR
   ********************************************************************/
  const NOTE_NAMES_CHORD = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

  const CHORD_FORMULAS = [
    { suffix: '', intervals: [0, 4, 7] },         // Major
    { suffix: 'm', intervals: [0, 3, 7] },        // Minor
    { suffix: 'dim', intervals: [0, 3, 6] },      // Diminished
    { suffix: 'aug', intervals: [0, 4, 8] },      // Augmented
    { suffix: 'sus4', intervals: [0, 5, 7] },     // Sus4
    { suffix: 'sus2', intervals: [0, 2, 7] },     // Sus2
    { suffix: '7', intervals: [0, 4, 7, 10] },    // Dominant 7th
    { suffix: 'maj7', intervals: [0, 4, 7, 11] }, // Major 7th
    { suffix: 'm7', intervals: [0, 3, 7, 10] },   // Minor 7th
    { suffix: 'dim7', intervals: [0, 3, 6, 9] },  // Diminished 7th
    { suffix: 'm7b5', intervals: [0, 3, 6, 10] }, // Half-Diminished
    { suffix: '6', intervals: [0, 4, 7, 9] },     // Major 6th
    { suffix: 'm6', intervals: [0, 3, 7, 9] },    // Minor 6th
    { suffix: 'add9', intervals: [0, 2, 4, 7] },  // Add9
    { suffix: '9', intervals: [0, 2, 4, 7, 10] }, // Dominant 9th
    { suffix: '5', intervals: [0, 7] }            // Power chord
  ];

  function detectChordSymbol(midiList) {
    if (!Array.isArray(midiList) || midiList.length === 0) return null;
    const sorted = Array.from(midiList).sort(function (a, b) { return a - b; });
    const bassMidi = sorted[0];
    const bassNoteName = NOTE_NAMES_CHORD[bassMidi % 12];

    const uniquePcs = Array.from(new Set(sorted.map(function (m) { return m % 12; })));
    if (uniquePcs.length === 1) {
      return {
        symbol: bassNoteName,
        notes: bassNoteName
      };
    }

    const notesStr = uniquePcs.map(function (pc) { return NOTE_NAMES_CHORD[pc]; }).join(' • ');

    let bestMatch = null;

    for (let rIdx = 0; rIdx < uniquePcs.length; rIdx++) {
      const root = uniquePcs[rIdx];
      const relIntervals = new Set(uniquePcs.map(function (pc) { return (pc - root + 12) % 12; }));

      for (let f = 0; f < CHORD_FORMULAS.length; f++) {
        const formula = CHORD_FORMULAS[f];
        let hasAll = true;
        for (let iv = 0; iv < formula.intervals.length; iv++) {
          if (!relIntervals.has(formula.intervals[iv])) {
            hasAll = false;
            break;
          }
        }
        if (hasAll) {
          const isExact = (relIntervals.size === formula.intervals.length);
          const isRootBass = (root === (bassMidi % 12));
          const score = (isExact ? 100 : 0) + formula.intervals.length * 10 + (isRootBass ? 5 : 0);
          if (!bestMatch || score > bestMatch.score) {
            bestMatch = {
              root: root,
              suffix: formula.suffix,
              score: score,
              isExact: isExact
            };
          }
        }
      }
    }

    if (bestMatch) {
      const rootName = NOTE_NAMES_CHORD[bestMatch.root];
      let sym = rootName + bestMatch.suffix;
      if (bestMatch.root !== (bassMidi % 12)) {
        sym += '/' + bassNoteName;
      }
      return {
        symbol: sym,
        notes: notesStr
      };
    }

    return {
      symbol: bassNoteName,
      notes: notesStr
    };
  }

  let lastChordSymbol = '';

  function updateLiveChordUI() {
    const chordBadge = document.getElementById('chordBadge');
    if (!chordBadge) return;
    if (!state.showChordDetector || state.viewMode !== 'play') {
      chordBadge.style.display = 'none';
      return;
    }

    const activeMidis = new Set();
    Object.keys(state.userPressed).forEach(function (mStr) {
      activeMidis.add(parseInt(mStr, 10));
    });
    if (state.isPlaying && state.song && Array.isArray(state.song.notes)) {
      const curT = state.currentTime;
      for (let i = 0; i < state.song.notes.length; i++) {
        const n = state.song.notes[i];
        if (n.time > curT + 0.05) break;
        if (n.time <= curT + 0.04 && (n.time + n.duration) >= curT - 0.02) {
          const handActive = !state.handIsoMaster ||
            (n.hand === 'L' && state.leftHandOn) ||
            (n.hand === 'R' && state.rightHandOn);
          if (handActive) activeMidis.add(n.midi);
        }
      }
    }

    if (activeMidis.size === 0) {
      chordBadge.style.display = 'none';
      lastChordSymbol = '';
      return;
    }

    const chord = detectChordSymbol(Array.from(activeMidis));
    if (!chord) {
      chordBadge.style.display = 'none';
      lastChordSymbol = '';
      return;
    }

    const symEl = document.getElementById('chordSymbolText');
    const notesEl = document.getElementById('chordNotesText');
    if (symEl) symEl.textContent = chord.symbol;
    if (notesEl) notesEl.textContent = chord.notes;

    if (chord.symbol !== lastChordSymbol) {
      lastChordSymbol = chord.symbol;
      chordBadge.classList.remove('pop');
      void chordBadge.offsetWidth;
      chordBadge.classList.add('pop');
    }
    chordBadge.style.display = 'flex';
  }

  /********************************************************************
   * USER INPUT: TOUCH/MOUSE PIANO, COMPUTER KEYBOARD & WEBMIDI
   ********************************************************************/
  function userNoteOn(midi, vel = 0.82) {
    const hand = (midi < 60) ? 'L' : 'R';
    state.userPressed[midi] = { hand: hand, time: performance.now() };
    audioEngine.playNote(midi, 0.75, vel);

    const kGeom = state.keyMap[midi];
    if (kGeom) spawnParticles(kGeom.laneX, kGeom.laneW, hand);

    // Live Practice Hit / Wrong Key Tracking
    if (state.viewMode !== 'edit' && (state.isPlaying || (state.waitMode && state.waitingMidis.size > 0))) {
      let isHit = false;
      let isAlreadyScoredChordNote = false;

      if (state.waitMode && state.waitingMidis.has(midi)) {
        isHit = true;
      }
      if (state.song && Array.isArray(state.song.notes)) {
        const curT = state.currentTime;
        const tol = 0.28;
        for (let i = 0; i < state.song.notes.length; i++) {
          const n = state.song.notes[i];
          if (n.time > curT + tol) break;
          const handActive = !state.handIsoMaster ||
            (n.hand === 'L' && state.leftHandOn) ||
            (n.hand === 'R' && state.rightHandOn);
          if (!handActive) continue;

          if (n.midi === midi) {
            if (n.time >= curT - tol) {
              if (!state.hitScoredNoteIndices.has(i)) {
                state.hitScoredNoteIndices.add(i);
                isHit = true;
                break;
              }
            }
            if (!isHit && state.hitScoredNoteIndices.has(i) && n.time <= curT + tol && (n.time + n.duration) >= curT - 0.15) {
              isAlreadyScoredChordNote = true;
            }
          }
        }
      }

      if (isHit || !isAlreadyScoredChordNote) {
        if (state.stats.hits + state.stats.wrongKeys === 0 && state.song && Array.isArray(state.song.notes)) {
          const prePassCutoff = state.currentTime - 0.30;
          for (let i = 0; i < state.song.notes.length; i++) {
            if (state.song.notes[i].time >= prePassCutoff) break;
            state.missedNoteIndices.add(i);
          }
        }
      }

      if (isHit) {
        state.stats.hits++;
        state.stats.streak++;
        if (state.stats.streak > state.stats.bestStreak) state.stats.bestStreak = state.stats.streak;
        updateAccuracyUI();
      } else if (!isAlreadyScoredChordNote) {
        state.stats.wrongKeys++;
        state.passWrongKeys++;
        state.stats.streak = 0;
        state.wrongKeyFlashes[midi] = performance.now();
        if (kGeom) spawnParticles(kGeom.laneX, kGeom.laneW, 'wrong');
        updateAccuracyUI();
      }
    }

    if (state.waitMode && state.waitingMidis.has(midi)) {
      if (state.waitStartedAtMs !== null && (performance.now() - state.waitStartedAtMs) > 3000) {
        state.passHadLongWaitStall = true;
      }
      state.waitingMidis.delete(midi);
      Array.from(state.waitingNoteIndices).forEach(function (idx) {
        const wn = state.song && state.song.notes ? state.song.notes[idx] : null;
        if (wn && wn.midi === midi) {
          state.playedIndices.add(idx);
          state.clearedWaitNoteIndices.add(idx);
          state.waitingNoteIndices.delete(idx);
        }
      });
      if (state.waitingMidis.size === 0) {
        state.waitStartedAtMs = null;
      }
      updateWaitBanner();
    }
  }

  function userNoteOff(midi) {
    delete state.userPressed[midi];
  }

  function hitTestPiano(x, y) {
    if (y < strikeY - 10) return null;
    for (let i = 0; i < state.blackKeys.length; i++) {
      const bk = state.blackKeys[i];
      if (x >= bk.x && x <= bk.x + bk.w && y >= bk.y && y <= bk.y + bk.h) return bk.midi;
    }
    for (let i = 0; i < state.whiteKeys.length; i++) {
      const wk = state.whiteKeys[i];
      if (x >= wk.x && x <= wk.x + wk.w && y >= wk.y && y <= wk.y + wk.h) return wk.midi;
    }
    return null;
  }

  /********************************************************************
   * PHASE 4A: VISUAL NOTE EDITOR HELPERS, UNDO/REDO & CANVAS GESTURES
   ********************************************************************/
  function xToNearestMidi(x) {
    for (let i = 0; i < state.blackKeys.length; i++) {
      const bk = state.blackKeys[i];
      if (x >= bk.x && x <= bk.x + bk.w) return bk.midi;
    }
    for (let i = 0; i < state.whiteKeys.length; i++) {
      const wk = state.whiteKeys[i];
      if (x >= wk.x && x <= wk.x + wk.w) return wk.midi;
    }
    let bestMidi = state.minMidi;
    let bestDist = Infinity;
    for (let i = 0; i < state.whiteKeys.length; i++) {
      const wk = state.whiteKeys[i];
      const cx = wk.x + wk.w * 0.5;
      const d = Math.abs(x - cx);
      if (d < bestDist) {
        bestDist = d;
        bestMidi = wk.midi;
      }
    }
    return bestMidi;
  }

  function snapTimeToGrid(sec) {
    const clamped = Math.max(0, sec);
    if (!state.editorSnapBeats || state.editorSnapBeats <= 0) {
      return Math.round(clamped * 1000) / 1000;
    }
    const bpmVal = Math.max(30, Math.min(260, state.song.bpm || 120));
    const beatSec = 60 / bpmVal;
    const stepSec = beatSec * state.editorSnapBeats;
    const offset = typeof state.song.gridOffset === 'number' ? state.song.gridOffset : 0.8;
    const snapped = offset + Math.round((clamped - offset) / stepSec) * stepSec;
    return Math.max(0, Math.round(snapped * 1000) / 1000);
  }

  function hitTestRunwayNote(x, y) {
    if (y >= strikeY || y < 0) return null;
    const { songHorizon, pxPerSongSec } = getRunwayTiming();
    const curT = state.currentTime;
    const notes = state.song.notes;

    for (let i = notes.length - 1; i >= 0; i--) {
      const n = notes[i];
      const nStart = n.time;
      const nEnd = n.time + n.duration;
      if (nEnd < curT - 0.04 || nStart > curT + songHorizon + 0.1) continue;

      const kGeom = state.keyMap[n.midi];
      if (!kGeom) continue;

      const rawBotY = strikeY - (nStart - curT) * pxPerSongSec;
      const rawTopY = strikeY - (nEnd - curT) * pxPerSongSec;
      const botY = Math.min(strikeY, rawBotY);
      const topY = Math.max(-16, rawTopY);
      if (botY - topY <= 2) continue;

      if (x >= kGeom.laneX - 4 && x <= kGeom.laneX + kGeom.laneW + 4 && y >= topY - 8 && y <= botY + 4) {
        if (Math.abs(y - topY) <= 12) {
          return { index: i, zone: 'resize-top' };
        }
        return { index: i, zone: 'body' };
      }
    }
    return null;
  }

  function cloneSongSnapshot(song) {
    return {
      notes: (song.notes || []).map(function (n) {
        return {
          midi: n.midi,
          name: n.name || midiToName(n.midi),
          time: n.time,
          duration: n.duration,
          hand: n.hand || (n.midi < 60 ? 'L' : 'R'),
          velocity: n.velocity || 0.78,
          finger: (n.finger >= 1 && n.finger <= 5) ? n.finger : undefined
        };
      }),
      bpm: song.bpm || 120,
      gridOffset: typeof song.gridOffset === 'number' ? song.gridOffset : 0.8,
      beatsPerBar: song.beatsPerBar || 4,
      beatType: song.beatType || 4,
      duration: song.duration || 15
    };
  }

  function syncUndoRedoUI() {
    const uBtn = document.getElementById('editorUndoBtn');
    const rBtn = document.getElementById('editorRedoBtn');
    if (uBtn) uBtn.disabled = state.undoStack.length === 0;
    if (rBtn) rBtn.disabled = state.redoStack.length === 0;
  }

  function syncEditorInspectorUI() {
    const inspectorBar = document.getElementById('noteInspectorBar');
    const labelEl = document.getElementById('inspectorNoteLabel');
    const mergeBtn = document.getElementById('inspectorMergeBtn');
    const cloneBtn = document.getElementById('inspectorCloneBtn');
    const handBtn = document.getElementById('inspectorHandBtn');
    const semiDownBtn = document.getElementById('inspectorSemiDownBtn');
    const semiUpBtn = document.getElementById('inspectorSemiUpBtn');
    const octDownBtn = document.getElementById('inspectorOctDownBtn');
    const octUpBtn = document.getElementById('inspectorOctUpBtn');
    const delBtn = document.getElementById('inspectorDeleteNoteBtn');
    const bpmInput = document.getElementById('editorBpmInput');

    if (bpmInput && document.activeElement !== bpmInput) {
      bpmInput.value = Math.round(state.song.bpm || 120);
    }

    // Prune any out-of-range indices from selectedNoteIndices
    if (state.selectedNoteIndices && state.selectedNoteIndices.size > 0) {
      Array.from(state.selectedNoteIndices).forEach(function (i) {
        if (i < 0 || i >= state.song.notes.length) {
          state.selectedNoteIndices.delete(i);
        }
      });
    }

    const groupCount = state.selectedNoteIndices ? state.selectedNoteIndices.size : 0;
    if (groupCount > 0 && (state.selectedNoteIndex < 0 || !state.selectedNoteIndices.has(state.selectedNoteIndex))) {
      const arr = Array.from(state.selectedNoteIndices);
      state.selectedNoteIndex = arr[arr.length - 1];
    }

    const idx = state.selectedNoteIndex;
    const note = (idx >= 0 && idx < state.song.notes.length) ? state.song.notes[idx] : null;
    const hasSelection = !!(note || groupCount > 0);
    if (inspectorBar) {
      inspectorBar.classList.toggle('visible', hasSelection);
    }

    if (!hasSelection) {
      state.isCloningNote = false;
      if (labelEl) {
        labelEl.classList.remove('suspect-active');
        if (state.editorTool === 'group') {
          labelEl.textContent = 'Group Select: Tap any notes or drag a box to select multiple notes';
        } else {
          labelEl.textContent = 'Tap a note to inspect/edit • Drag empty runway to scroll time';
        }
      }
      if (mergeBtn) {
        mergeBtn.disabled = true;
        mergeBtn.textContent = '🔗 Merge Selected';
      }
      if (cloneBtn) {
        cloneBtn.disabled = true;
        cloneBtn.classList.remove('active');
        cloneBtn.textContent = '⧉ Copy to Key';
      }
      if (handBtn) {
        handBtn.disabled = true;
        handBtn.textContent = 'Hand: R';
      }
      if (semiDownBtn) semiDownBtn.disabled = true;
      if (semiUpBtn) semiUpBtn.disabled = true;
      if (octDownBtn) octDownBtn.disabled = true;
      if (octUpBtn) octUpBtn.disabled = true;
      if (delBtn) delBtn.disabled = true;
      document.querySelectorAll('.finger-opt-btn').forEach(function (fBtn) {
        fBtn.disabled = true;
        fBtn.classList.remove('active');
      });
      return;
    }

    if (groupCount >= 2) {
      const selectedArr = Array.from(state.selectedNoteIndices);
      const firstNote = state.song.notes[selectedArr[0]] || note;
      const firstMidi = firstNote ? firstNote.midi : null;
      const allSamePitch = selectedArr.every(function (idx) {
        return state.song.notes[idx] && state.song.notes[idx].midi === firstMidi;
      });
      const pitchName = firstNote ? (firstNote.name || midiToName(firstNote.midi)) : '';
      if (mergeBtn) {
        mergeBtn.disabled = !allSamePitch;
        mergeBtn.textContent = '🔗 Merge Selected (' + groupCount + ')';
      }
      if (labelEl) {
        labelEl.classList.remove('suspect-active');
        if (allSamePitch) {
          labelEl.textContent = groupCount + ' × ' + pitchName + ' notes selected — ready to merge, flip hand, or edit';
        } else {
          labelEl.textContent = groupCount + ' notes selected across multiple keys — ready to flip hand, transpose, snap, or delete';
        }
      }
    } else {
      if (mergeBtn) {
        mergeBtn.disabled = true;
        mergeBtn.textContent = '🔗 Merge Selected';
      }
      if (labelEl && note) {
        if (state.isCloningNote) {
          labelEl.classList.remove('suspect-active');
          labelEl.textContent = 'Copy Mode: Tap any piano key or runway lane to place copy';
        } else if (state.editorTool === 'group') {
          labelEl.classList.remove('suspect-active');
          labelEl.textContent = '1 note (' + note.name + ') selected — tap more notes or drag a box to select multiple';
        } else {
          const fingerSuffix = (note.finger >= 1 && note.finger <= 5) ? (' • Finger: ' + note.finger) : '';
          let text = note.name + ' (MIDI ' + note.midi + ')' + fingerSuffix + ' • Start: ' + note.time.toFixed(2) + 's • Len: ' + note.duration.toFixed(2) + 's';
          if (note.suspect && note.suspectReason) {
            text += ' • ⚠ ' + note.suspectReason;
          }
          labelEl.textContent = text;
          labelEl.classList.toggle('suspect-active', !!(note && note.suspect));
        }
      }
    }

    if (cloneBtn) {
      cloneBtn.disabled = !note;
      cloneBtn.classList.toggle('active', !!state.isCloningNote);
      cloneBtn.textContent = state.isCloningNote ? 'Tap Target Key...' : '⧉ Copy to Key';
    }
    if (handBtn && note) {
      handBtn.disabled = false;
      handBtn.textContent = 'Hand: ' + (note.hand === 'L' ? 'L (Blue)' : 'R (Green)');
    }
    if (semiDownBtn) semiDownBtn.disabled = !note;
    if (semiUpBtn) semiUpBtn.disabled = !note;
    if (octDownBtn) octDownBtn.disabled = !note;
    if (octUpBtn) octUpBtn.disabled = !note;
    if (delBtn) delBtn.disabled = !note && groupCount === 0;

    const curFinger = (note && note.finger >= 1 && note.finger <= 5) ? note.finger : 0;
    document.querySelectorAll('.finger-opt-btn').forEach(function (fBtn) {
      fBtn.disabled = false;
      const fNum = parseInt(fBtn.getAttribute('data-finger'), 10);
      fBtn.classList.toggle('active', fNum === curFinger);
    });
  }

  function setFingerForSelectedNotes(fingerNum) {
    if (!state.song || !Array.isArray(state.song.notes)) return;
    const targets = [];
    if (state.selectedNoteIndices && state.selectedNoteIndices.size > 0) {
      state.selectedNoteIndices.forEach(function (idx) {
        if (state.song.notes[idx]) targets.push(state.song.notes[idx]);
      });
    } else if (state.selectedNoteIndex >= 0 && state.song.notes[state.selectedNoteIndex]) {
      targets.push(state.song.notes[state.selectedNoteIndex]);
    }
    if (targets.length === 0) return;

    pushUndoSnapshot();
    const fVal = (fingerNum >= 1 && fingerNum <= 5) ? fingerNum : undefined;
    targets.forEach(function (n) {
      n.finger = fVal;
    });

    commitSongEdit(null);
    syncEditorInspectorUI();
  }

  function pushUndoSnapshot() {
    state.undoStack.push(cloneSongSnapshot(state.song));
    if (state.undoStack.length > 30) {
      state.undoStack.shift();
    }
    state.redoStack = [];
    syncUndoRedoUI();
  }

  let autoSaveFlashTimer = null;
  function flashAutoSaveIndicator(saveSuccess) {
    const bannerEl = document.getElementById('editAutoSaveBanner');
    const dotEl = document.getElementById('editAutoSaveDot');
    const textEl = document.getElementById('editAutoSaveText');
    if (!bannerEl || !textEl) return;
    if (saveSuccess === false) {
      if (autoSaveFlashTimer) clearTimeout(autoSaveFlashTimer);
      bannerEl.classList.remove('saving');
      if (dotEl) dotEl.style.background = '#ef4444';
      textEl.textContent = '⚠ Save failed (storage full)';
      return;
    }
    if (dotEl) {
      dotEl.className = 'autosave-dot';
      dotEl.style.background = '';
    }
    bannerEl.classList.add('saving');
    textEl.textContent = 'Saving changes...';
    if (autoSaveFlashTimer) clearTimeout(autoSaveFlashTimer);
    autoSaveFlashTimer = setTimeout(function () {
      bannerEl.classList.remove('saving');
      textEl.textContent = '✓ Changes saved automatically';
    }, 350);
  }

  function persistActiveSong() {
    if (!state.song) return;
    if (!state.song.id) state.song.id = state.activeSongId || ('custom_' + Date.now());
    if (!state.song.isCustom) {
      state.song.isModifiedBuiltin = true;
    }
    if (audioTranscriber && audioTranscriber.customSongStore) {
      flashAutoSaveIndicator();
      Promise.resolve(audioTranscriber.customSongStore.saveSong(state.song))
        .then(function (ok) {
          if (ok === false) flashAutoSaveIndicator(false);
        })
        .catch(function () {
          flashAutoSaveIndicator(false);
        });
    } else {
      flashAutoSaveIndicator(false);
    }
  }

  function analyzeSongConfidence(song) {
    const suspectGroup = document.getElementById('editorSuspectGroup');
    const toggleBtn = document.getElementById('editorSuspectToggleBtn');
    const nextBtn = document.getElementById('editorNextSuspectBtn');
    const updateSuspectButtons = function (count) {
      if (suspectGroup) suspectGroup.style.display = count > 0 ? 'flex' : 'none';
      if (toggleBtn) toggleBtn.textContent = '⚠ Suspect (' + count + ')';
      if (nextBtn) nextBtn.disabled = (count === 0);
    };

    if (!song || !Array.isArray(song.notes) || song.notes.length === 0) {
      updateSuspectButtons(0);
      return 0;
    }

    if (!song.isCustom && !song.isModifiedBuiltin) {
      for (let i = 0; i < song.notes.length; i++) {
        song.notes[i].suspect = false;
        song.notes[i].suspectReason = '';
      }
      updateSuspectButtons(0);
      return 0;
    }

    const pcWeights = new Array(12).fill(0);
    const pcCounts = new Array(12).fill(0);
    const velocities = [];
    const midis = [];

    for (let i = 0; i < song.notes.length; i++) {
      const n = song.notes[i];
      const pc = ((n.midi % 12) + 12) % 12;
      const vel = typeof n.velocity === 'number' ? n.velocity : 0.78;
      const dur = typeof n.duration === 'number' ? n.duration : 0.25;
      pcWeights[pc] += dur * vel;
      pcCounts[pc]++;
      velocities.push(vel);
      midis.push(n.midi);
    }

    const MAJOR_OFFSETS = [0, 2, 4, 5, 7, 9, 11];
    const MINOR_OFFSETS = [0, 2, 3, 5, 7, 8, 10];
    let bestRoot = 0;
    let bestMode = 'Major';
    let bestScore = -1;
    let bestOffsets = MAJOR_OFFSETS;

    for (let r = 0; r < 12; r++) {
      const modes = [
        { name: 'Major', offsets: MAJOR_OFFSETS },
        { name: 'Minor', offsets: MINOR_OFFSETS }
      ];
      for (let m = 0; m < modes.length; m++) {
        const offsets = modes[m].offsets;
        let score = 0;
        for (let k = 0; k < offsets.length; k++) {
          const pc = (r + offsets[k]) % 12;
          const mult = (offsets[k] === 0 || offsets[k] === 7) ? 1.05 : 1.0;
          score += pcWeights[pc] * mult;
        }
        if (score > bestScore) {
          bestScore = score;
          bestRoot = r;
          bestMode = modes[m].name;
          bestOffsets = offsets;
        }
      }
    }

    song.detectedKeyName = NOTE_NAMES_CHORD[bestRoot] + ' ' + bestMode;
    const inScale = new Set(bestOffsets.map(function (off) { return (bestRoot + off) % 12; }));
    if (bestMode === 'Minor') {
      const raised7 = (bestRoot + 11) % 12;
      if (pcCounts[raised7] >= 3) {
        inScale.add(raised7);
      }
    }

    velocities.sort(function (a, b) { return a - b; });
    midis.sort(function (a, b) { return a - b; });
    const medVel = velocities[Math.floor(velocities.length * 0.5)] || 0.78;
    const p12Vel = velocities[Math.floor(velocities.length * 0.12)] || 0.25;
    const p05Midi = midis[Math.floor(midis.length * 0.05)] || 36;
    const p95Midi = midis[Math.floor(midis.length * 0.95)] || 84;

    let suspectCount = 0;
    for (let i = 0; i < song.notes.length; i++) {
      const n = song.notes[i];
      const reasons = [];
      const vel = typeof n.velocity === 'number' ? n.velocity : 0.78;
      const dur = typeof n.duration === 'number' ? n.duration : 0.25;
      const pc = ((n.midi % 12) + 12) % 12;

      if (vel <= Math.min(0.30, p12Vel)) {
        reasons.push('Quiet ghost note (' + Math.round(vel * 100) + '% vel)');
      }
      if (dur < 0.085) {
        reasons.push('Short micro-note (' + Math.round(dur * 1000) + 'ms)');
      }
      if (!inScale.has(pc) && pcCounts[pc] <= Math.max(2, Math.floor(song.notes.length * 0.015)) && (vel < medVel * 0.82 || dur < 0.14)) {
        reasons.push('Out-of-key outlier (' + song.detectedKeyName + ')');
      }
      if ((n.midi < p05Midi - 12 || n.midi > p95Midi + 12) && vel < medVel * 0.85) {
        reasons.push('Extreme register outlier');
      }
      // Check overlap / double-trigger on the same MIDI key within 40ms
      for (let j = i - 1; j >= 0; j--) {
        const prev = song.notes[j];
        if (n.time - prev.time > 0.25) break;
        if (prev.midi === n.midi && (n.time - prev.time) < 0.04) {
          reasons.push('Double-triggered overlap');
          break;
        }
      }
      n.suspect = reasons.length > 0;
      n.suspectReason = reasons.join(' • ');
      if (n.suspect) suspectCount++;
    }

    updateSuspectButtons(suspectCount);
    return suspectCount;
  }

  function commitSongEdit(keepNoteRef, options) {
    if (!state.song || !Array.isArray(state.song.notes)) return;

    const keptGroupRefs = [];
    if (!keepNoteRef) {
      if (state.selectedNoteIndices && state.selectedNoteIndices.size > 0) {
        state.selectedNoteIndices.forEach(function (i) {
          if (i >= 0 && i < state.song.notes.length) {
            keptGroupRefs.push(state.song.notes[i]);
          }
        });
      } else if (state.selectedNoteIndex >= 0 && state.selectedNoteIndex < state.song.notes.length) {
        keptGroupRefs.push(state.song.notes[state.selectedNoteIndex]);
      }
    }

    state.song.notes.sort(function (a, b) {
      return (a.time - b.time) || (a.midi - b.midi);
    });

    if (keepNoteRef) {
      state.selectedNoteIndex = state.song.notes.indexOf(keepNoteRef);
      state.selectedNoteIndices.clear();
      if (state.selectedNoteIndex >= 0) {
        state.selectedNoteIndices.add(state.selectedNoteIndex);
      }
    } else if (keptGroupRefs.length > 0) {
      state.selectedNoteIndices.clear();
      keptGroupRefs.forEach(function (ref) {
        const newIdx = state.song.notes.indexOf(ref);
        if (newIdx >= 0) state.selectedNoteIndices.add(newIdx);
      });
      if (state.selectedNoteIndices.size > 0) {
        const arr = Array.from(state.selectedNoteIndices);
        state.selectedNoteIndex = arr[arr.length - 1];
      } else {
        state.selectedNoteIndex = -1;
      }
    } else if (state.selectedNoteIndex >= state.song.notes.length || state.selectedNoteIndex < 0) {
      state.selectedNoteIndex = -1;
      state.selectedNoteIndices.clear();
    }

    let maxEnd = 0;
    for (let i = 0; i < state.song.notes.length; i++) {
      const end = state.song.notes[i].time + state.song.notes[i].duration;
      if (end > maxEnd) maxEnd = end;
    }
    state.song.duration = Math.max(10, Math.round((maxEnd + 2.0) * 100) / 100);
    const shouldRebuildBeats = (options && options.rebuildBeats) ||
      !Array.isArray(state.song.beats) ||
      state.song.beats.length === 0;
    if (shouldRebuildBeats) {
      const bpmVal = Math.max(30, Math.min(260, state.song.bpm || 120));
      const offsetVal = typeof state.song.gridOffset === 'number' ? state.song.gridOffset : 0.8;
      const bpbVal = state.song.beatsPerBar || 4;
      const beatTypeVal = state.song.beatType || 4;
      state.song.beats = generateBeatGrid(state.song.duration, (60 / bpmVal) * (4 / beatTypeVal), bpbVal, offsetVal);
    }

    persistActiveSong();
    resetNoteTriggersBefore(state.currentTime);
    calculateAutoRange(state);
    rebuildKeyboardGeometry();
    syncTimelineUI();
    syncLoopUI();
    analyzeSongConfidence(state.song);
    syncEditorInspectorUI();
    syncUndoRedoUI();
  }

  function applySongSnapshot(snap) {
    if (!snap || !state.song) return;
    state.song.notes = snap.notes.map(function (n) {
      return {
        midi: n.midi,
        name: n.name || midiToName(n.midi),
        time: n.time,
        duration: n.duration,
        hand: n.hand,
        velocity: n.velocity,
        finger: (n.finger >= 1 && n.finger <= 5) ? n.finger : undefined
      };
    });
    state.song.bpm = snap.bpm || 120;
    state.song.gridOffset = typeof snap.gridOffset === 'number' ? snap.gridOffset : 0.8;
    state.song.beatsPerBar = snap.beatsPerBar || 4;
    if (snap.beatType) state.song.beatType = snap.beatType;
    state.song.duration = snap.duration || 15;
    state.selectedNoteIndex = -1;
    state.selectedNoteIndices.clear();
    state.groupDragBox = null;
    commitSongEdit(null, { rebuildBeats: true });
  }

  function undoEdit() {
    if (state.undoStack.length === 0) return;
    state.redoStack.push(cloneSongSnapshot(state.song));
    const prev = state.undoStack.pop();
    applySongSnapshot(prev);
  }

  function redoEdit() {
    if (state.redoStack.length === 0) return;
    state.undoStack.push(cloneSongSnapshot(state.song));
    const next = state.redoStack.pop();
    applySongSnapshot(next);
  }

  function updateGroupDragSelection(box, baseIndices) {
    if (!box || !state.song || !Array.isArray(state.song.notes)) return;
    const minX = Math.min(box.x0, box.x1);
    const maxX = Math.max(box.x0, box.x1);
    const minY = Math.min(box.y0, box.y1);
    const maxY = Math.max(box.y0, box.y1);

    const { songHorizon, pxPerSongSec } = getRunwayTiming();
    const curT = state.currentTime;
    const notes = state.song.notes;

    const intersecting = [];
    for (let i = 0; i < notes.length; i++) {
      const n = notes[i];
      const nStart = n.time;
      const nEnd = n.time + n.duration;
      if (nEnd < curT - 0.04 || nStart > curT + songHorizon + 0.1) continue;

      const kGeom = state.keyMap[n.midi];
      if (!kGeom) continue;

      const rawBotY = strikeY - (nStart - curT) * pxPerSongSec;
      const rawTopY = strikeY - (nEnd - curT) * pxPerSongSec;
      const botY = Math.min(strikeY, rawBotY);
      const topY = Math.max(-16, rawTopY);
      if (botY - topY <= 2) continue;

      const nx0 = kGeom.laneX + 1.5;
      const nx1 = nx0 + Math.max(8, kGeom.laneW - 3);
      if (nx1 >= minX && nx0 <= maxX && botY >= minY && topY <= maxY) {
        intersecting.push({ index: i, midi: n.midi });
      }
    }

    state.selectedNoteIndices = new Set(baseIndices || []);
    for (let k = 0; k < intersecting.length; k++) {
      state.selectedNoteIndices.add(intersecting[k].index);
    }

    if (state.selectedNoteIndices.size > 0) {
      const arr = Array.from(state.selectedNoteIndices);
      state.selectedNoteIndex = arr[arr.length - 1];
    } else {
      state.selectedNoteIndex = -1;
    }
    syncEditorInspectorUI();
  }

  canvas.addEventListener('pointerdown', function (e) {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const hitKey = hitTestPiano(x, y);

    // Edit Mode Copy-to-Key (Chord / Interval Builder): works on both virtual piano keys and runway lanes
    if (state.viewMode === 'edit' && state.isCloningNote && state.selectedNoteIndex >= 0 && state.selectedNoteIndex < state.song.notes.length) {
      if (hitKey !== null || y < strikeY) {
        const srcNote = state.song.notes[state.selectedNoteIndex];
        const targetMidi = hitKey !== null ? hitKey : xToNearestMidi(x);
        pushUndoSnapshot();

        let targetRef = null;
        for (let i = 0; i < state.song.notes.length; i++) {
          const existing = state.song.notes[i];
          if (existing.midi === targetMidi && Math.abs(existing.time - srcNote.time) < 0.01) {
            existing.duration = srcNote.duration;
            existing.hand = srcNote.hand || (targetMidi < 60 ? 'L' : 'R');
            existing.velocity = srcNote.velocity || 0.8;
            targetRef = existing;
            break;
          }
        }

        if (!targetRef) {
          targetRef = {
            midi: targetMidi,
            name: midiToName(targetMidi),
            time: srcNote.time,
            duration: srcNote.duration,
            hand: srcNote.hand || (targetMidi < 60 ? 'L' : 'R'),
            velocity: srcNote.velocity || 0.8
          };
          state.song.notes.push(targetRef);
        }

        const previewDur = Math.min(0.9, Math.max(0.25, srcNote.duration));
        audioEngine.playNote(srcNote.midi, previewDur, srcNote.velocity || 0.8);
        if (targetMidi !== srcNote.midi) {
          audioEngine.playNote(targetMidi, previewDur, srcNote.velocity || 0.8);
        }

        state.isCloningNote = false;
        commitSongEdit(targetRef);
        return;
      }
    }

    if (hitKey !== null) {
      try { canvas.setPointerCapture(e.pointerId); } catch (err) {}
      pointerMidis[e.pointerId] = hitKey;
      userNoteOn(hitKey, 0.85);
      return;
    }

    if (y >= strikeY) return;

    // Play Mode: clicking the runway toggles Play/Pause
    if (state.viewMode !== 'edit') {
      setPlaying(!state.isPlaying);
      return;
    }

    // Edit Mode Runway Interaction
    try { canvas.setPointerCapture(e.pointerId); } catch (err) {}
    const { pxPerSongSec: pxPerSec } = getRunwayTiming();
    const hitNote = hitTestRunwayNote(x, y);

    if (state.editorTool === 'delete') {
      if (hitNote) {
        pushUndoSnapshot();
        state.song.notes.splice(hitNote.index, 1);
        state.selectedNoteIndex = -1;
        state.selectedNoteIndices.clear();
        commitSongEdit(null);
      }
      return;
    }

    function createNoteAtPoint(ptX, ptY) {
      const midi = xToNearestMidi(ptX);
      const rawT = state.currentTime + (strikeY - ptY) / pxPerSec;
      const snappedT = snapTimeToGrid(rawT);
      pushUndoSnapshot();
      const newNote = {
        midi: midi,
        name: midiToName(midi),
        time: snappedT,
        duration: 0.35,
        hand: midi < 60 ? 'L' : 'R',
        velocity: 0.82
      };
      state.song.notes.push(newNote);
      state.selectedNoteIndex = state.song.notes.length - 1;
      state.selectedNoteIndices.clear();
      state.selectedNoteIndices.add(state.selectedNoteIndex);
      audioEngine.playNote(midi, 0.35, 0.82);
      syncEditorInspectorUI();
      editDrag = {
        pointerId: e.pointerId,
        mode: 'resize',
        noteRef: newNote,
        origTime: newNote.time,
        origDur: newNote.duration,
        origMidi: newNote.midi,
        startX: ptX,
        startY: ptY,
        undoPushed: true,
        modified: true
      };
    }

    if (state.editorTool === 'add') {
      createNoteAtPoint(x, y);
      return;
    }

    // Group Select Mode (or Shift+Click in Select Mode)
    const isGroupAction = (state.editorTool === 'group') || (state.editorTool === 'select' && e.shiftKey);
    if (isGroupAction) {
      if (hitNote) {
        const tappedIdx = hitNote.index;
        const tappedNote = state.song.notes[tappedIdx];
        if (!tappedNote) return;

        // Seed from single selectedNoteIndex if set is empty
        if (state.selectedNoteIndices.size === 0 && state.selectedNoteIndex >= 0 && state.selectedNoteIndex < state.song.notes.length) {
          if (state.selectedNoteIndex !== tappedIdx) {
            state.selectedNoteIndices.add(state.selectedNoteIndex);
          }
        }

        if (state.selectedNoteIndices.has(tappedIdx)) {
          state.selectedNoteIndices.delete(tappedIdx);
          if (state.selectedNoteIndices.size > 0) {
            const rem = Array.from(state.selectedNoteIndices);
            state.selectedNoteIndex = rem[rem.length - 1];
          } else {
            state.selectedNoteIndex = -1;
          }
        } else {
          state.selectedNoteIndices.add(tappedIdx);
          state.selectedNoteIndex = tappedIdx;
          audioEngine.playNote(tappedNote.midi, 0.35, tappedNote.velocity || 0.8);
        }
        syncEditorInspectorUI();
        return;
      }

      // Empty runway in Group Select mode: start Marquee drag box
      state.groupDragBox = { x0: x, y0: y, x1: x, y1: y };
      editDrag = {
        pointerId: e.pointerId,
        mode: 'group-box',
        startX: x,
        startY: y,
        baseIndices: new Set(state.selectedNoteIndices)
      };
      return;
    }

    // Single Select Tool Mode
    if (hitNote) {
      state.selectedNoteIndex = hitNote.index;
      state.selectedNoteIndices.clear();
      state.selectedNoteIndices.add(hitNote.index);
      const note = state.song.notes[hitNote.index];
      audioEngine.playNote(note.midi, 0.35, note.velocity || 0.8);
      syncEditorInspectorUI();
      editDrag = {
        pointerId: e.pointerId,
        mode: hitNote.zone === 'resize-top' ? 'resize' : 'move',
        noteRef: note,
        origTime: note.time,
        origDur: note.duration,
        origMidi: note.midi,
        startX: x,
        startY: y,
        undoPushed: false,
        modified: false
      };
    } else {
      const nowMs = performance.now();
      const dtTap = nowMs - lastEmptyRunwayTap.timeMs;
      const distTap = Math.hypot(x - lastEmptyRunwayTap.x, y - lastEmptyRunwayTap.y);
      if (dtTap < 320 && distTap < 20) {
        lastEmptyRunwayTap.timeMs = 0;
        createNoteAtPoint(x, y);
        return;
      }
      lastEmptyRunwayTap = { timeMs: nowMs, x: x, y: y };
      state.selectedNoteIndex = -1;
      state.selectedNoteIndices.clear();
      syncEditorInspectorUI();
      editDrag = {
        pointerId: e.pointerId,
        mode: 'scroll',
        startY: y,
        origCurrentTime: state.currentTime
      };
    }
  });

  canvas.addEventListener('pointermove', function (e) {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (pointerMidis[e.pointerId] !== undefined) {
      const hit = hitTestPiano(x, y);
      const prev = pointerMidis[e.pointerId];
      if (hit !== prev) {
        if (prev !== null) userNoteOff(prev);
        pointerMidis[e.pointerId] = hit;
        if (hit !== null) userNoteOn(hit, 0.76);
      }
      return;
    }

    if (editDrag && editDrag.pointerId === e.pointerId) {
      const { pxPerSongSec: pxPerSec } = getRunwayTiming();

      if (editDrag.mode === 'group-box') {
        state.groupDragBox = {
          x0: editDrag.startX,
          y0: editDrag.startY,
          x1: x,
          y1: Math.min(strikeY, Math.max(0, y))
        };
        updateGroupDragSelection(state.groupDragBox, editDrag.baseIndices);
        return;
      }

      if (editDrag.mode === 'scroll') {
        const dy = y - editDrag.startY;
        const dtSec = dy / pxPerSec;
        seekTo(editDrag.origCurrentTime + dtSec);
        return;
      }

      const note = editDrag.noteRef;
      if (!note) return;

      if (editDrag.mode === 'move') {
        const dx = x - editDrag.startX;
        const dy = y - editDrag.startY;
        if (Math.abs(dx) < 3 && Math.abs(dy) < 3 && !editDrag.modified) return;

        if (!editDrag.undoPushed) {
          pushUndoSnapshot();
          editDrag.undoPushed = true;
        }
        editDrag.modified = true;

        const newMidi = xToNearestMidi(x);
        if (newMidi !== note.midi) {
          note.midi = newMidi;
          note.name = midiToName(newMidi);
          audioEngine.playNote(newMidi, 0.3, note.velocity || 0.8);
        }

        if (Math.abs(dy) >= 10) {
          const rawNewTime = editDrag.origTime - (dy / pxPerSec);
          note.time = snapTimeToGrid(rawNewTime);
        } else {
          note.time = editDrag.origTime;
        }
        syncEditorInspectorUI();
      } else if (editDrag.mode === 'resize') {
        if (!editDrag.undoPushed) {
          pushUndoSnapshot();
          editDrag.undoPushed = true;
        }
        editDrag.modified = true;

        const rawEndTime = state.currentTime + (strikeY - y) / pxPerSec;
        const snappedEnd = snapTimeToGrid(rawEndTime);
        const newDur = Math.max(0.06, Math.min(8.0, snappedEnd - note.time));
        note.duration = Math.round(newDur * 1000) / 1000;
        syncEditorInspectorUI();
      }
    }
  });

  function releasePointer(e) {
    if (pointerMidis[e.pointerId] !== undefined) {
      if (pointerMidis[e.pointerId] !== null) {
        userNoteOff(pointerMidis[e.pointerId]);
      }
      delete pointerMidis[e.pointerId];
    }
    if (editDrag && editDrag.pointerId === e.pointerId) {
      const finishedDrag = editDrag;
      editDrag = null;
      if (finishedDrag.mode === 'group-box') {
        if (state.groupDragBox) {
          const dist = Math.hypot(
            state.groupDragBox.x1 - state.groupDragBox.x0,
            state.groupDragBox.y1 - state.groupDragBox.y0
          );
          if (dist < 5) {
            state.selectedNoteIndices.clear();
            state.selectedNoteIndex = -1;
          } else {
            updateGroupDragSelection(state.groupDragBox, finishedDrag.baseIndices);
          }
        }
        state.groupDragBox = null;
        syncEditorInspectorUI();
        return;
      }
      if (finishedDrag.modified && finishedDrag.noteRef) {
        commitSongEdit(finishedDrag.noteRef);
      }
    }
  }
  canvas.addEventListener('pointerup', releasePointer);
  canvas.addEventListener('pointercancel', releasePointer);

  // Mouse Wheel / Trackpad Scrolling on Runway in Edit Mode
  canvas.addEventListener('wheel', function (e) {
    if (state.viewMode === 'edit') {
      e.preventDefault();
      const pxPerSec = Math.max(40, getRunwayTiming().pxPerSongSec);
      seekTo(state.currentTime - (e.deltaY / pxPerSec));
    }
  }, { passive: false });

  // Computer Keyboard Support & Edit Mode Shortcuts
  const KEY_OFFSETS = { a: 0, w: 1, s: 2, e: 3, d: 4, f: 5, t: 6, g: 7, y: 8, h: 9, u: 10, j: 11, k: 12, o: 13, l: 14 };
  window.addEventListener('keydown', function (e) {
    if (e.target && (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT')) return;
    if (state.viewMode === 'home') return;

    const k = String(e.key || '').toLowerCase();

    // Edit Mode Undo / Redo & Note Deletion / Transpose Shortcuts
    if (state.viewMode === 'edit') {
      if ((e.ctrlKey || e.metaKey) && k === 'z') {
        e.preventDefault();
        if (e.shiftKey) redoEdit();
        else undoEdit();
        return;
      }
      if ((e.ctrlKey || e.metaKey) && k === 'y') {
        e.preventDefault();
        redoEdit();
        return;
      }
      if ((e.key === 'Delete' || e.key === 'Backspace') && (state.selectedNoteIndex >= 0 || state.selectedNoteIndices.size > 0)) {
        e.preventDefault();
        pushUndoSnapshot();
        if (state.selectedNoteIndices.size > 0) {
          state.song.notes = state.song.notes.filter(function (_, i) {
            return !state.selectedNoteIndices.has(i);
          });
        } else {
          state.song.notes.splice(state.selectedNoteIndex, 1);
        }
        state.selectedNoteIndex = -1;
        state.selectedNoteIndices.clear();
        commitSongEdit(null);
        return;
      }
      if (!e.ctrlKey && !e.metaKey && !e.altKey && e.key >= '0' && e.key <= '5' && (state.selectedNoteIndex >= 0 || state.selectedNoteIndices.size > 0)) {
        e.preventDefault();
        setFingerForSelectedNotes(parseInt(e.key, 10));
        return;
      }
      if (!e.ctrlKey && !e.metaKey && !e.altKey && k === 'n') {
        e.preventDefault();
        jumpToNextSuspectNote();
        return;
      }
    }

    if (e.code === 'Space') {
      e.preventDefault();
      if (state.viewMode === 'edit') {
        setViewMode('play');
      }
      setPlaying(!state.isPlaying);
      return;
    }
    if (!e.ctrlKey && !e.metaKey && !e.altKey && (k === 'q' || k === 'm')) {
      setHudMinimized(!state.hudMinimized);
      return;
    }
    if (!e.ctrlKey && !e.metaKey && !e.altKey && KEY_OFFSETS[k] !== undefined && !heldKeys[k]) {
      heldKeys[k] = true;
      userNoteOn(60 + KEY_OFFSETS[k], 0.82);
    }
  });

  window.addEventListener('keyup', function (e) {
    const k = String(e.key || '').toLowerCase();
    if (KEY_OFFSETS[k] !== undefined) {
      delete heldKeys[k];
      userNoteOff(60 + KEY_OFFSETS[k]);
    }
  });

  // WebMIDI Initialization, Sustain Pedal (CC 64), and Live Connection Badge
  const midiStatusBadge = document.getElementById('midiStatusBadge');
  const midiStatusText = document.getElementById('midiStatusText');

  function updateMidiBadge(connectedNames) {
    if (!midiStatusBadge || !midiStatusText) return;
    if (connectedNames && connectedNames.length > 0) {
      const primaryName = connectedNames[0];
      midiStatusBadge.classList.add('connected');
      midiStatusText.textContent = 'MIDI: ' + primaryName;
      midiStatusBadge.title = 'Connected MIDI: ' + connectedNames.join(', ');
    } else {
      midiStatusBadge.classList.remove('connected', 'sustain-active');
      midiStatusText.textContent = 'MIDI';
      midiStatusBadge.title = 'Connect a USB or Bluetooth MIDI keyboard';
    }
  }

  try {
    if (navigator.requestMIDIAccess) {
      navigator.requestMIDIAccess().then(function (access) {
        function bindInputs() {
          const connectedNames = [];
          access.inputs.forEach(function (input) {
            if (input.state === 'connected') {
              connectedNames.push(input.name || input.manufacturer || 'Keyboard');
            }
            input.onmidimessage = function (msg) {
              if (!msg.data || msg.data.length < 2) return;
              const cmd = msg.data[0] & 0xf0;
              const data1 = msg.data[1];
              const data2 = msg.data.length > 2 ? msg.data[2] : 0;

              if (cmd === 0x90 && data2 > 0) {
                userNoteOn(data1, data2 / 127);
              } else if (cmd === 0x80 || (cmd === 0x90 && data2 === 0)) {
                userNoteOff(data1);
              } else if (cmd === 0xb0 && data1 === 64) {
                // Control Change 64: Sustain Pedal
                const pedalDown = data2 >= 64;
                audioEngine.setSustainPedal(pedalDown);
                if (midiStatusBadge) {
                  midiStatusBadge.classList.toggle('sustain-active', pedalDown);
                }
              }
            };
          });
          updateMidiBadge(connectedNames);
        }
        bindInputs();
        access.onstatechange = bindInputs;
      }).catch(function () {});
    }
  } catch (e) {}

  /********************************************************************
   * PHASE 4A: HOME LIBRARY, VIEW SWITCHER & EDITOR TOOLBAR BINDINGS
   ********************************************************************/
  const homeView = document.getElementById('homeView');
  const homeSearchInput = document.getElementById('homeSearchInput');
  const homeSongGrid = document.getElementById('homeSongGrid');
  const homeViewCardsBtn = document.getElementById('homeViewCardsBtn');
  const homeViewCompactBtn = document.getElementById('homeViewCompactBtn');
  const homeOpenAiModalBtn = document.getElementById('homeOpenAiModalBtn');
  const homeAddSongBtn = document.getElementById('homeAddSongBtn');
  const addSongModal = document.getElementById('addSongModal');
  const closeAddSongModalBtn = document.getElementById('closeAddSongModalBtn');
  const homeOpenMidiBtn = document.getElementById('homeOpenMidiBtn');
  const backToHomeBtn = document.getElementById('backToHomeBtn');
  const toggleEditModeBtn = document.getElementById('toggleEditModeBtn');
  const editorToolbar = document.getElementById('editorToolbar');
  const hudPanel = document.getElementById('hudPanel');
  const hudToggleBtn = document.getElementById('hudToggleBtn');
  const hudMiniSummary = document.getElementById('hudMiniSummary');

  function syncHomeLayoutToggleUI() {
    const isCompact = state.homeLayout === 'compact';
    if (homeSongGrid) {
      homeSongGrid.classList.toggle('layout-compact', isCompact);
      homeSongGrid.classList.toggle('layout-cards', !isCompact);
    }
    if (homeViewCardsBtn) {
      homeViewCardsBtn.classList.toggle('active', !isCompact);
    }
    if (homeViewCompactBtn) {
      homeViewCompactBtn.classList.toggle('active', isCompact);
    }
  }

  function setHomeLayout(layout, savePref) {
    state.homeLayout = (layout === 'compact') ? 'compact' : 'cards';
    if (savePref) {
      try {
        localStorage.setItem('keyflow_home_layout', state.homeLayout);
      } catch (e) {}
    }
    syncHomeLayoutToggleUI();
    renderHomeSongGrid();
  }

  function downloadSongMidi(songObj) {
    if (!songObj || !audioTranscriber) return;
    try {
      const blob = audioTranscriber.exportSongToMidiBlob(songObj);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      const safeName = String(songObj.title || 'KeyFlow_Song')
        .replace(/[^a-zA-Z0-9_\-\s]/g, '')
        .trim()
        .replace(/\s+/g, '_') || 'KeyFlow_Song';
      a.href = url;
      a.download = safeName + '.mid';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(function () { URL.revokeObjectURL(url); }, 3000);
    } catch (err) {
      console.warn('MIDI Export error:', err);
    }
  }

  function deleteOrResetSong(id) {
    const s = SONGS[id];
    if (!s) return;
    if (s.isCustom) {
      if (!window.confirm('Delete "' + (s.title || 'this song') + '" from your library? This cannot be undone.')) {
        return;
      }
      if (audioTranscriber && audioTranscriber.customSongStore) {
        audioTranscriber.customSongStore.deleteSong(id);
      }
      delete SONGS[id];
      if (state.activeSongId === id) {
        loadSongById('let_it_be');
      }
      renderHomeSongGrid();
    } else if (FACTORY_BUILTIN_SONGS[id]) {
      if (!window.confirm('Reset "' + (s.title || 'this song') + '" back to factory default? Your edits will be lost.')) {
        return;
      }
      SONGS[id] = JSON.parse(JSON.stringify(FACTORY_BUILTIN_SONGS[id]));
      if (audioTranscriber && audioTranscriber.customSongStore) {
        audioTranscriber.customSongStore.deleteSong(id);
      }
      if (state.activeSongId === id) {
        loadSongById(id);
      }
      renderHomeSongGrid();
    }
  }

  function getSongSourceCategory(songObj) {
    if (!songObj) return 'builtin';
    const src = String(songObj.source || '').toLowerCase();
    if (src.indexOf('sheet') !== -1 || src.indexOf('musicxml') !== -1 || src.indexOf('mxl') !== -1 || src.indexOf('omr') !== -1) return 'sheet';
    if (src.indexOf('ai') !== -1) return 'ai';
    if (src.indexOf('midi') !== -1) return 'midi';
    if (songObj.isCustom) return 'midi';
    return 'builtin';
  }

  function computeSongRangeLabel(notes) {
    if (!Array.isArray(notes) || notes.length === 0) return '';
    let minM = 127;
    let maxM = 0;
    for (let i = 0; i < notes.length; i++) {
      const m = notes[i] && notes[i].midi;
      if (typeof m === 'number') {
        if (m < minM) minM = m;
        if (m > maxM) maxM = m;
      }
    }
    if (minM > maxM) return '';
    const fmt = function (midi) {
      const n = NOTE_NAMES_CHORD[((midi % 12) + 12) % 12];
      const oct = Math.floor(midi / 12) - 1;
      return n + oct;
    };
    return fmt(minM) + '–' + fmt(maxM);
  }

  function estimateSongDifficulty(song) {
    const notes = song && Array.isArray(song.notes) ? song.notes : [];
    const dur = Math.max(1, song && song.duration ? song.duration : 1);
    const nps = notes.length / dur;
    if (nps < 2.2) return { label: 'Easy', cls: 'diff-easy' };
    if (nps < 4.5) return { label: 'Intermediate', cls: 'diff-mid' };
    return { label: 'Challenging', cls: 'diff-hard' };
  }

  function renderHomeSongGrid() {
    if (!homeSongGrid) return;
    syncHomeLayoutToggleUI();
    homeSongGrid.innerHTML = '';
    const bestAccMap = loadBestAccuracyMap();

    const allIds = Object.keys(SONGS);
    const customList = [];
    const builtinList = [];

    for (let i = 0; i < allIds.length; i++) {
      const id = allIds[i];
      const s = SONGS[id];
      if (!s || !Array.isArray(s.notes)) continue;
      if (!s.id) s.id = id;
      if (s.isCustom) {
        customList.push(s);
      } else {
        builtinList.push(s);
      }
    }

    customList.sort(function (a, b) {
      return (b.createdAt || 0) - (a.createdAt || 0);
    });

    const orderedSongs = customList.concat(builtinList);
    const query = String(state.homeSearch || '').trim().toLowerCase();
    const filter = state.homeFilter || 'all';

    const matchedSongs = orderedSongs.filter(function (s) {
      const cat = getSongSourceCategory(s);
      if (filter !== 'all' && cat !== filter) return false;
      if (query && String(s.title || '').toLowerCase().indexOf(query) === -1) return false;
      return true;
    });

    if (matchedSongs.length === 0) {
      const emptyEl = document.createElement('div');
      emptyEl.className = 'song-grid-empty';
      emptyEl.textContent = 'No songs match this filter. Upload a .MID file or convert an MP3 song with + Add Song!';
      homeSongGrid.appendChild(emptyEl);
      return;
    }

    matchedSongs.forEach(function (s) {
      const cat = getSongSourceCategory(s);
      const card = document.createElement('div');
      card.className = 'song-card';

      // Compact Top Row: Clickable Song Title (expands details) + Play Button
      const mainRow = document.createElement('div');
      mainRow.className = 'song-card-main';

      const titleToggle = document.createElement('div');
      titleToggle.className = 'song-card-title-toggle';
      titleToggle.title = state.homeLayout === 'compact'
        ? 'Tap to expand song details & options'
        : s.title;

      const chevronEl = document.createElement('span');
      chevronEl.className = 'song-card-chevron';
      chevronEl.textContent = '▸';

      const titleEl = document.createElement('h3');
      titleEl.className = 'song-card-title';
      titleEl.textContent = s.title;

      titleToggle.appendChild(chevronEl);
      titleToggle.appendChild(titleEl);
      titleToggle.addEventListener('click', function () {
        if (state.homeLayout === 'compact') {
          card.classList.toggle('expanded');
        }
      });

      const playBtn = document.createElement('button');
      playBtn.type = 'button';
      playBtn.className = 'ui-btn ui-btn-emerald song-card-play-btn';
      playBtn.textContent = '▶ Play';
      playBtn.addEventListener('click', function (e) {
        e.stopPropagation();
        loadSongById(s.id);
        setViewMode('play');
      });

      mainRow.appendChild(titleToggle);
      mainRow.appendChild(playBtn);

      // Collapsible Details Section: Source Badges, Metadata & Secondary Actions
      const detailsDiv = document.createElement('div');
      detailsDiv.className = 'song-card-details';

      const badgesRow = document.createElement('div');
      badgesRow.className = 'song-card-badges';

      const srcBadge = document.createElement('span');
      if (cat === 'sheet') {
        srcBadge.className = 'song-badge sheet';
        srcBadge.textContent = 'Sheet Music';
      } else if (cat === 'ai') {
        srcBadge.className = 'song-badge ai';
        srcBadge.textContent = 'AI Audio';
      } else if (cat === 'midi') {
        srcBadge.className = 'song-badge midi';
        srcBadge.textContent = 'MIDI Upload';
      } else {
        srcBadge.className = 'song-badge builtin';
        srcBadge.textContent = 'Built-In Classic';
      }
      badgesRow.appendChild(srcBadge);

      if (s.isModifiedBuiltin) {
        const editBadge = document.createElement('span');
        editBadge.className = 'song-badge edited';
        editBadge.textContent = 'Edited';
        badgesRow.appendChild(editBadge);
      }

      const diff = estimateSongDifficulty(s);
      const diffBadge = document.createElement('span');
      diffBadge.className = 'song-badge ' + diff.cls;
      diffBadge.textContent = diff.label;
      badgesRow.appendChild(diffBadge);

      if (typeof bestAccMap[s.id] === 'number') {
        const accBadge = document.createElement('span');
        accBadge.className = 'song-badge acc-best';
        accBadge.textContent = 'Best ' + bestAccMap[s.id] + '%';
        badgesRow.appendChild(accBadge);
      }

      const metaEl = document.createElement('div');
      metaEl.className = 'song-card-meta';
      const bpmNum = Math.round(s.bpm || 120);
      const rangeLabel = computeSongRangeLabel(s.notes);
      metaEl.textContent = formatTime(s.duration)
        + ' • ' + s.notes.length + ' notes'
        + ' • ' + bpmNum + ' BPM'
        + (rangeLabel ? ' • ' + rangeLabel : '');

      const actionsRow = document.createElement('div');
      actionsRow.className = 'song-card-actions';

      const editBtn = document.createElement('button');
      editBtn.type = 'button';
      editBtn.className = 'ui-btn ui-btn-accent';
      editBtn.textContent = '✎ Edit Notes';
      editBtn.addEventListener('click', function () {
        loadSongById(s.id);
        setViewMode('edit');
      });

      const midiBtn = document.createElement('button');
      midiBtn.type = 'button';
      midiBtn.className = 'ui-btn';
      midiBtn.title = 'Export 2-Track Standard .MID File';
      midiBtn.textContent = '⬇ .MID';
      midiBtn.addEventListener('click', function () {
        downloadSongMidi(s);
      });

      actionsRow.appendChild(editBtn);
      actionsRow.appendChild(midiBtn);

      if (s.isCustom || s.isModifiedBuiltin) {
        const delBtn = document.createElement('button');
        delBtn.type = 'button';
        delBtn.className = 'ui-btn ui-btn-danger';
        delBtn.title = s.isCustom ? 'Delete custom song from library' : 'Reset built-in song to factory default';
        delBtn.textContent = s.isCustom ? '🗑 Delete' : '↺ Reset';
        delBtn.addEventListener('click', function () {
          deleteOrResetSong(s.id);
        });
        actionsRow.appendChild(delBtn);
      }

      detailsDiv.appendChild(badgesRow);
      detailsDiv.appendChild(metaEl);
      detailsDiv.appendChild(actionsRow);

      card.appendChild(mainRow);
      card.appendChild(detailsDiv);
      homeSongGrid.appendChild(card);
    });
  }

  function setViewMode(mode, songId) {
    clearAllActiveInputs();

    if (songId && SONGS[songId]) {
      loadSongById(songId);
    }
    state.viewMode = mode;
    state.isCloningNote = false;
    state.groupDragBox = null;
    if (mode !== 'edit') {
      state.selectedNoteIndices.clear();
      state.selectedNoteIndex = -1;
    }

    if (homeView) {
      homeView.classList.toggle('hidden', mode !== 'home');
    }

    const editAutoSaveBanner = document.getElementById('editAutoSaveBanner');
    const editAutoSaveText = document.getElementById('editAutoSaveText');
    const waitBannerEl = document.getElementById('waitBanner');
    if (editAutoSaveBanner) {
      if (mode === 'edit') {
        editAutoSaveBanner.style.display = 'inline-flex';
        editAutoSaveBanner.classList.remove('saving');
        if (editAutoSaveText) editAutoSaveText.textContent = '✓ Changes saved automatically';
      } else {
        editAutoSaveBanner.style.display = 'none';
      }
    }
    if (waitBannerEl && mode === 'edit') {
      waitBannerEl.classList.remove('visible');
    }

    if (mode === 'home') {
      setPlaying(false);
      renderHomeSongGrid();
      return;
    }

    if (mode === 'edit') {
      setPlaying(false);
      if (hudPanel) hudPanel.style.display = 'none';
      if (editorToolbar) editorToolbar.style.display = 'flex';
      if (toggleEditModeBtn) {
        toggleEditModeBtn.textContent = '▶ Back to Play';
        toggleEditModeBtn.className = 'ui-btn ui-btn-emerald';
        toggleEditModeBtn.title = 'Return to Play Mode at current position';
      }
      syncEditorInspectorUI();
      syncUndoRedoUI();
      rebuildKeyboardGeometry();
    } else {
      // 'play' mode
      if (editorToolbar) editorToolbar.style.display = 'none';
      if (hudPanel) hudPanel.style.display = '';
      if (toggleEditModeBtn) {
        toggleEditModeBtn.textContent = '✎ Edit Notes';
        toggleEditModeBtn.className = 'ui-btn ui-btn-accent';
        toggleEditModeBtn.title = 'Open Visual Note Editor';
      }
      resetNoteTriggersBefore(state.currentTime);
      rebuildKeyboardGeometry();
    }
  }

  function loadSongById(id) {
    const s = SONGS[id];
    if (!s) return;
    setPlaying(false);
    cancelCountIn();
    if (audioEngine && typeof audioEngine.stopAll === 'function') {
      audioEngine.stopAll();
    }
    resetAccuracyStats();
    state.activeSongId = id;
    state.song = s;
    if (!s.bpm) s.bpm = 120;
    if (typeof s.gridOffset !== 'number') s.gridOffset = 0.8;
    if (!s.beatsPerBar) s.beatsPerBar = 4;

    state.currentTime = 0;
    state.loopA = Math.min(2.0, s.duration * 0.12);
    state.loopB = Math.max(state.loopA + 2.0, Math.min(s.duration * 0.48, Math.max(state.loopA + 2.0, s.duration - 0.5)));
    state.loopPassCount = 1;
    state.passHadLongWaitStall = false;
    state.passWrongKeys = 0;
    state.passMisses = 0;
    state.waitStartedAtMs = null;

    state.selectedNoteIndex = -1;
    state.selectedNoteIndices.clear();
    state.groupDragBox = null;
    state.isCloningNote = false;
    state.undoStack = [];
    state.redoStack = [];

    document.getElementById('songTitleHeader').textContent = s.title;

    const delBtn = document.getElementById('deleteCustomSongBtn');
    if (delBtn) {
      delBtn.style.display = (s.isCustom || s.isModifiedBuiltin) ? 'inline-flex' : 'none';
      const span = delBtn.querySelector('span') || delBtn;
      span.textContent = s.isCustom ? 'Delete Song' : 'Reset Song';
    }

    resetNoteTriggersBefore(0);
    calculateAutoRange(state);
    rebuildKeyboardGeometry();
    syncTimelineUI();
    syncLoopUI();
    analyzeSongConfidence(state.song);
    syncEditorInspectorUI();
    syncUndoRedoUI();
  }

  // Home View & Top Bar Navigation Listeners
  if (backToHomeBtn) {
    backToHomeBtn.addEventListener('click', function () {
      setViewMode('home');
    });
  }

  if (toggleEditModeBtn) {
    toggleEditModeBtn.addEventListener('click', function () {
      if (state.viewMode === 'edit') {
        setViewMode('play');
      } else {
        setViewMode('edit');
      }
    });
  }

  if (homeSearchInput) {
    homeSearchInput.addEventListener('input', function (e) {
      state.homeSearch = e.target.value || '';
      renderHomeSongGrid();
    });
  }

  document.querySelectorAll('.home-filter-btn').forEach(function (btn) {
    btn.addEventListener('click', function () {
      state.homeFilter = btn.getAttribute('data-filter') || 'all';
      document.querySelectorAll('.home-filter-btn').forEach(function (b) {
        b.classList.toggle('active', b === btn);
      });
      renderHomeSongGrid();
    });
  });

  if (homeViewCardsBtn) {
    homeViewCardsBtn.addEventListener('click', function () {
      setHomeLayout('cards', true);
    });
  }
  if (homeViewCompactBtn) {
    homeViewCompactBtn.addEventListener('click', function () {
      setHomeLayout('compact', true);
    });
  }

  // Visual Note Editor Toolbar Listeners
  const toolSelectBtn = document.getElementById('toolSelectBtn');
  const toolGroupBtn = document.getElementById('toolGroupBtn');
  const toolAddBtn = document.getElementById('toolAddBtn');
  const toolDeleteBtn = document.getElementById('toolDeleteBtn');
  const editorSnapSelect = document.getElementById('editorSnapSelect');
  const editorUndoBtn = document.getElementById('editorUndoBtn');
  const editorRedoBtn = document.getElementById('editorRedoBtn');
  const editorBpmInput = document.getElementById('editorBpmInput');
  const editorGridLeftBtn = document.getElementById('editorGridLeftBtn');
  const editorGridRightBtn = document.getElementById('editorGridRightBtn');
  const inspectorMergeBtn = document.getElementById('inspectorMergeBtn');
  const inspectorCloneBtn = document.getElementById('inspectorCloneBtn');
  const inspectorHandBtn = document.getElementById('inspectorHandBtn');
  const inspectorSemiDownBtn = document.getElementById('inspectorSemiDownBtn');
  const inspectorSemiUpBtn = document.getElementById('inspectorSemiUpBtn');
  const inspectorOctDownBtn = document.getElementById('inspectorOctDownBtn');
  const inspectorOctUpBtn = document.getElementById('inspectorOctUpBtn');
  const inspectorDeleteNoteBtn = document.getElementById('inspectorDeleteNoteBtn');

  function setEditorTool(toolName) {
    state.editorTool = toolName;
    state.isCloningNote = false;
    state.groupDragBox = null;
    [toolSelectBtn, toolGroupBtn, toolAddBtn, toolDeleteBtn].forEach(function (btn) {
      if (!btn) return;
      btn.classList.toggle('active', btn.getAttribute('data-tool') === toolName);
    });
    syncEditorInspectorUI();
  }

  if (toolSelectBtn) {
    toolSelectBtn.addEventListener('click', function () { setEditorTool('select'); });
  }
  if (toolGroupBtn) {
    toolGroupBtn.addEventListener('click', function () { setEditorTool('group'); });
  }
  if (toolAddBtn) {
    toolAddBtn.addEventListener('click', function () { setEditorTool('add'); });
  }
  if (toolDeleteBtn) {
    toolDeleteBtn.addEventListener('click', function () { setEditorTool('delete'); });
  }

  if (editorSnapSelect) {
    editorSnapSelect.addEventListener('change', function (e) {
      state.editorSnapBeats = parseFloat(e.target.value) || 0;
    });
  }

  if (editorUndoBtn) {
    editorUndoBtn.addEventListener('click', function () { undoEdit(); });
  }
  if (editorRedoBtn) {
    editorRedoBtn.addEventListener('click', function () { redoEdit(); });
  }

  if (editorBpmInput) {
    editorBpmInput.addEventListener('change', function (e) {
      const val = Math.max(40, Math.min(240, parseInt(e.target.value, 10) || 120));
      e.target.value = val;
      pushUndoSnapshot();
      state.song.bpm = val;
      commitSongEdit(null, { rebuildBeats: true });
    });
  }

  if (editorGridLeftBtn) {
    editorGridLeftBtn.addEventListener('click', function () {
      pushUndoSnapshot();
      const curOff = typeof state.song.gridOffset === 'number' ? state.song.gridOffset : 0.8;
      state.song.gridOffset = Math.max(0, Math.round((curOff - 0.05) * 1000) / 1000);
      commitSongEdit(null, { rebuildBeats: true });
    });
  }

  if (editorGridRightBtn) {
    editorGridRightBtn.addEventListener('click', function () {
      pushUndoSnapshot();
      const curOff = typeof state.song.gridOffset === 'number' ? state.song.gridOffset : 0.8;
      state.song.gridOffset = Math.round((curOff + 0.05) * 1000) / 1000;
      commitSongEdit(null, { rebuildBeats: true });
    });
  }

  function jumpToNextSuspectNote() {
    if (!state.song || !Array.isArray(state.song.notes) || state.song.notes.length === 0) return;
    const notes = state.song.notes;
    let startIdx = 0;
    if (state.selectedNoteIndex >= 0 && state.selectedNoteIndex < notes.length) {
      startIdx = (state.selectedNoteIndex + 1) % notes.length;
    } else {
      for (let i = 0; i < notes.length; i++) {
        if (notes[i].time >= state.currentTime) {
          startIdx = i;
          break;
        }
      }
    }

    let foundIdx = -1;
    for (let offset = 0; offset < notes.length; offset++) {
      const idx = (startIdx + offset) % notes.length;
      if (notes[idx].suspect === true) {
        foundIdx = idx;
        break;
      }
    }
    if (foundIdx < 0) return;

    const target = notes[foundIdx];
    state.selectedNoteIndex = foundIdx;
    state.selectedNoteIndices.clear();
    state.selectedNoteIndices.add(foundIdx);
    seekTo(Math.max(0, target.time - 0.75));
    audioEngine.playNote(target.midi, Math.min(0.6, target.duration), target.velocity || 0.78);
    syncEditorInspectorUI();
  }

  const editorSuspectToggleBtn = document.getElementById('editorSuspectToggleBtn');
  const editorNextSuspectBtn = document.getElementById('editorNextSuspectBtn');
  const editorSnapAllBtn = document.getElementById('editorSnapAllBtn');

  if (editorSuspectToggleBtn) {
    editorSuspectToggleBtn.addEventListener('click', function () {
      state.showSuspectNotes = !state.showSuspectNotes;
      editorSuspectToggleBtn.classList.toggle('active', state.showSuspectNotes);
    });
  }

  if (editorNextSuspectBtn) {
    editorNextSuspectBtn.addEventListener('click', function () {
      jumpToNextSuspectNote();
    });
  }

  if (editorSnapAllBtn) {
    editorSnapAllBtn.addEventListener('click', function () {
      if (!state.song || !Array.isArray(state.song.notes) || state.song.notes.length === 0) return;
      const snapBeats = state.editorSnapBeats > 0 ? state.editorSnapBeats : 0.25;
      const stepSec = (60 / Math.max(30, Math.min(260, state.song.bpm || 120))) * snapBeats;
      const offset = typeof state.song.gridOffset === 'number' ? state.song.gridOffset : 0.8;

      const hasSelection = state.selectedNoteIndices && state.selectedNoteIndices.size > 0;
      const candidates = [];
      if (hasSelection) {
        state.selectedNoteIndices.forEach(function (idx) {
          if (state.song.notes[idx]) candidates.push(state.song.notes[idx]);
        });
      } else {
        for (let i = 0; i < state.song.notes.length; i++) {
          candidates.push(state.song.notes[i]);
        }
      }

      const changes = [];
      for (let i = 0; i < candidates.length; i++) {
        const n = candidates[i];
        const nearest = offset + Math.round((n.time - offset) / stepSec) * stepSec;
        const diff = Math.abs(n.time - nearest);
        if (nearest >= 0 && diff <= stepSec * 0.38 && diff >= 0.004) {
          changes.push({ note: n, nextTime: Math.round(nearest * 1000) / 1000 });
        }
      }

      if (changes.length === 0) {
        const labelEl = document.getElementById('inspectorNoteLabel');
        if (labelEl) {
          labelEl.classList.remove('suspect-active');
          labelEl.textContent = '✓ Snapped 0 note(s) to beat grid';
        }
        return;
      }

      pushUndoSnapshot();
      for (let i = 0; i < changes.length; i++) {
        changes[i].note.time = changes[i].nextTime;
      }
      commitSongEdit(null);
      const labelEl = document.getElementById('inspectorNoteLabel');
      if (labelEl) {
        labelEl.classList.remove('suspect-active');
        labelEl.textContent = '✓ Snapped ' + changes.length + ' note(s) to beat grid';
      }
    });
  }

  const editorSplitHandSelect = document.getElementById('editorSplitHandSelect');
  const editorResplitHandsBtn = document.getElementById('editorResplitHandsBtn');

  if (editorResplitHandsBtn) {
    editorResplitHandsBtn.addEventListener('click', function () {
      if (!state.song || !Array.isArray(state.song.notes) || state.song.notes.length === 0) return;
      const modeVal = editorSplitHandSelect ? editorSplitHandSelect.value : 'auto';
      const splitMidi = (modeVal !== 'auto') ? parseInt(modeVal, 10) : null;

      pushUndoSnapshot();

      const hasMultiSelection = state.selectedNoteIndices && state.selectedNoteIndices.size >= 2;
      let targetNotes = state.song.notes;
      if (hasMultiSelection) {
        targetNotes = [];
        state.selectedNoteIndices.forEach(function (idx) {
          if (state.song.notes[idx]) targetNotes.push(state.song.notes[idx]);
        });
      }

      if (splitMidi !== null && !isNaN(splitMidi)) {
        for (let i = 0; i < targetNotes.length; i++) {
          targetNotes[i].hand = (targetNotes[i].midi < splitMidi) ? 'L' : 'R';
        }
      } else if (typeof assignHandsSmart === 'function') {
        assignHandsSmart(targetNotes);
      }

      let leftCount = 0;
      let rightCount = 0;
      for (let i = 0; i < targetNotes.length; i++) {
        if (targetNotes[i].hand === 'L') leftCount++;
        else rightCount++;
      }

      const keepRef = (!hasMultiSelection && state.selectedNoteIndex >= 0) ? state.song.notes[state.selectedNoteIndex] : null;
      commitSongEdit(keepRef);
      const labelEl = document.getElementById('inspectorNoteLabel');
      if (labelEl) {
        labelEl.classList.remove('suspect-active');
        labelEl.textContent = '✓ Hands split: ' + leftCount + ' Left (Blue) • ' + rightCount + ' Right (Green)';
      }
    });
  }

  const editorLegatoBtn = document.getElementById('editorLegatoBtn');
  if (editorLegatoBtn) {
    editorLegatoBtn.addEventListener('click', function () {
      if (!state.song || !Array.isArray(state.song.notes) || state.song.notes.length < 2) return;
      if (typeof extendChordLegato !== 'function') return;

      const hasMultiSelection = state.selectedNoteIndices && state.selectedNoteIndices.size >= 2;
      let targetNotes = state.song.notes;
      if (hasMultiSelection) {
        targetNotes = [];
        state.selectedNoteIndices.forEach(function (idx) {
          if (state.song.notes[idx]) targetNotes.push(state.song.notes[idx]);
        });
      }

      pushUndoSnapshot();
      const count = extendChordLegato(targetNotes);
      if (count === 0) {
        state.undoStack.pop();
        syncUndoRedoUI();
      }
      const keepRef = (!hasMultiSelection && state.selectedNoteIndex >= 0) ? state.song.notes[state.selectedNoteIndex] : null;
      commitSongEdit(keepRef);
      const labelEl = document.getElementById('inspectorNoteLabel');
      if (labelEl) {
        labelEl.classList.remove('suspect-active');
        labelEl.textContent = '✓ Extended legato sustain on ' + count + ' note(s)';
      }
    });
  }

  function transposeSelectedNote(semitones) {
    if (!state.song || !Array.isArray(state.song.notes)) return;
    const indices = (state.selectedNoteIndices && state.selectedNoteIndices.size > 0)
      ? Array.from(state.selectedNoteIndices)
      : (state.selectedNoteIndex >= 0 ? [state.selectedNoteIndex] : []);
    if (indices.length === 0) return;

    pushUndoSnapshot();
    let lastAuditionNote = null;
    indices.forEach(function (idx) {
      if (idx >= 0 && idx < state.song.notes.length) {
        const note = state.song.notes[idx];
        const nextMidi = Math.max(21, Math.min(108, note.midi + semitones));
        note.midi = nextMidi;
        note.name = midiToName(nextMidi);
        lastAuditionNote = note;
      }
    });

    if (lastAuditionNote) {
      audioEngine.playNote(lastAuditionNote.midi, 0.35, lastAuditionNote.velocity || 0.8);
      commitSongEdit(null);
    }
  }

  if (inspectorMergeBtn) {
    inspectorMergeBtn.addEventListener('click', function () {
      if (!state.selectedNoteIndices || state.selectedNoteIndices.size < 2) return;

      const selectedNotes = [];
      state.selectedNoteIndices.forEach(function (idx) {
        if (idx >= 0 && idx < state.song.notes.length) {
          selectedNotes.push(state.song.notes[idx]);
        }
      });

      if (selectedNotes.length < 2) return;

      const targetMidi = selectedNotes[0].midi;
      for (let i = 1; i < selectedNotes.length; i++) {
        if (selectedNotes[i].midi !== targetMidi) {
          const labelEl = document.getElementById('inspectorNoteLabel');
          if (labelEl) {
            labelEl.classList.add('suspect-active');
            labelEl.textContent = '⚠ Only notes of the same pitch can be merged (' + midiToName(targetMidi) + ' and ' + midiToName(selectedNotes[i].midi) + ' selected)';
          }
          return;
        }
      }

      pushUndoSnapshot();

      let startTime = Infinity;
      let endTime = -Infinity;
      let maxVel = 0;
      for (let i = 0; i < selectedNotes.length; i++) {
        const sn = selectedNotes[i];
        if (sn.time < startTime) startTime = sn.time;
        if (sn.time + sn.duration > endTime) endTime = sn.time + sn.duration;
        const v = sn.velocity || 0.78;
        if (v > maxVel) maxVel = v;
      }
      const mergedDuration = Math.max(0.06, Math.round((endTime - startTime) * 1000) / 1000);
      const hand = selectedNotes[0].hand || (targetMidi < 60 ? 'L' : 'R');

      let preservedFinger = undefined;
      for (let i = 0; i < selectedNotes.length; i++) {
        if (selectedNotes[i].finger >= 1 && selectedNotes[i].finger <= 5) {
          preservedFinger = selectedNotes[i].finger;
          break;
        }
      }

      const mergedNote = {
        midi: targetMidi,
        name: midiToName(targetMidi),
        time: startTime,
        duration: mergedDuration,
        hand: hand,
        velocity: maxVel,
        finger: preservedFinger
      };

      state.song.notes = state.song.notes.filter(function (_, idx) {
        return !state.selectedNoteIndices.has(idx);
      });
      state.song.notes.push(mergedNote);

      state.isCloningNote = false;
      state.selectedNoteIndices.clear();
      audioEngine.playNote(mergedNote.midi, Math.min(1.2, mergedNote.duration), mergedNote.velocity);
      commitSongEdit(mergedNote);
    });
  }

  if (inspectorCloneBtn) {
    inspectorCloneBtn.addEventListener('click', function () {
      const idx = state.selectedNoteIndex;
      if (idx < 0 || idx >= state.song.notes.length) {
        state.isCloningNote = false;
        syncEditorInspectorUI();
        return;
      }
      state.isCloningNote = !state.isCloningNote;
      syncEditorInspectorUI();
    });
  }

  if (inspectorHandBtn) {
    inspectorHandBtn.addEventListener('click', function () {
      if (!state.song || !Array.isArray(state.song.notes)) return;
      const indices = (state.selectedNoteIndices && state.selectedNoteIndices.size > 0)
        ? Array.from(state.selectedNoteIndices)
        : (state.selectedNoteIndex >= 0 ? [state.selectedNoteIndex] : []);
      if (indices.length === 0) return;

      pushUndoSnapshot();
      let targetHand = null;
      indices.forEach(function (idx) {
        if (idx >= 0 && idx < state.song.notes.length) {
          const note = state.song.notes[idx];
          if (targetHand === null) {
            targetHand = (note.hand === 'L') ? 'R' : 'L';
          }
          note.hand = targetHand;
        }
      });
      commitSongEdit(null);
    });
  }

  if (inspectorSemiDownBtn) {
    inspectorSemiDownBtn.addEventListener('click', function () { transposeSelectedNote(-1); });
  }
  if (inspectorSemiUpBtn) {
    inspectorSemiUpBtn.addEventListener('click', function () { transposeSelectedNote(1); });
  }
  if (inspectorOctDownBtn) {
    inspectorOctDownBtn.addEventListener('click', function () { transposeSelectedNote(-12); });
  }
  if (inspectorOctUpBtn) {
    inspectorOctUpBtn.addEventListener('click', function () { transposeSelectedNote(12); });
  }

  if (inspectorDeleteNoteBtn) {
    inspectorDeleteNoteBtn.addEventListener('click', function () {
      if (state.selectedNoteIndex < 0 && state.selectedNoteIndices.size === 0) return;
      pushUndoSnapshot();
      if (state.selectedNoteIndices.size > 0) {
        state.song.notes = state.song.notes.filter(function (_, idx) {
          return !state.selectedNoteIndices.has(idx);
        });
      } else {
        state.song.notes.splice(state.selectedNoteIndex, 1);
      }
      state.selectedNoteIndex = -1;
      state.selectedNoteIndices.clear();
      commitSongEdit(null);
    });
  }

  // Phase 4B: Finger Number Selector Buttons in Note Inspector
  document.querySelectorAll('.finger-opt-btn').forEach(function (fBtn) {
    fBtn.addEventListener('click', function (e) {
      e.stopPropagation();
      const f = parseInt(this.getAttribute('data-finger'), 10);
      setFingerForSelectedNotes(isNaN(f) ? 0 : f);
    });
  });

  document.getElementById('rangeModeSelect').addEventListener('change', function (e) {
    state.rangeMode = e.target.value;
    calculateAutoRange(state);
    rebuildKeyboardGeometry();
  });

  function importMidiFile(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function (ev) {
      try {
        const parsed = parseStandardMidiFile(ev.target.result, file.name);
        const id = 'midi_' + Date.now();
        parsed.id = id;
        parsed.isCustom = true;
        parsed.source = 'midi-upload';
        parsed.createdAt = Date.now();
        SONGS[id] = parsed;
        if (audioTranscriber && audioTranscriber.customSongStore) {
          audioTranscriber.customSongStore.saveSong(parsed);
        }
        renderHomeSongGrid();
        loadSongById(id);
        setViewMode('play');
      } catch (err) {
        console.warn('MIDI parse error:', err);
        alert('Could not load MIDI file: ' + (err && err.message ? err.message : 'Invalid or corrupted .mid file'));
      }
    };
    reader.readAsArrayBuffer(file);
  }

  document.getElementById('midiInput').addEventListener('change', function (e) {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    importMidiFile(file);
    e.target.value = '';
  });

  // Play / Pause & Restart
  document.getElementById('playPauseBtn').addEventListener('click', function () {
    if (state.viewMode === 'edit') {
      setViewMode('play');
    }
    setPlaying(!state.isPlaying);
  });
  document.getElementById('restartBtn').addEventListener('click', function () {
    seekTo(state.loopEnabled ? state.loopA : 0);
  });

  // Speed Selector & Collapsible HUD Panel
  function setHudMinimized(minimized) {
    state.hudMinimized = !!minimized;
    if (hudPanel) {
      hudPanel.classList.toggle('minimized', state.hudMinimized);
    }
    if (hudMiniSummary) {
      hudMiniSummary.textContent = formatSpeedBadge(state.speed);
    }
  }

  if (hudToggleBtn) {
    hudToggleBtn.addEventListener('click', function (e) {
      e.stopPropagation();
      setHudMinimized(!state.hudMinimized);
    });
  }
  if (hudPanel) {
    hudPanel.addEventListener('click', function () {
      if (state.hudMinimized) {
        setHudMinimized(false);
      }
    });
  }

  function applyPlaybackSpeed(nextSpeed) {
    const clamped = Math.max(0.25, Math.min(1.5, Math.round(nextSpeed * 20) / 20));
    state.speed = clamped;
    const speedStr = formatSpeedBadge(state.speed);
    const speedBtnEl = document.getElementById('speedBtn');
    if (speedBtnEl) speedBtnEl.textContent = speedStr;
    if (hudMiniSummary) hudMiniSummary.textContent = speedStr;
    syncLoopUI();
  }

  const speedBtnEl = document.getElementById('speedBtn');
  if (speedBtnEl) {
    speedBtnEl.addEventListener('click', function () {
      // Tap center pill: jump back to 1.0x if modified, or toggle to 0.75x practice speed
      const target = Math.abs(state.speed - 1.0) > 0.01 ? 1.0 : 0.75;
      applyPlaybackSpeed(target);
    });
  }

  const speedDownBtn = document.getElementById('speedDownBtn');
  if (speedDownBtn) {
    speedDownBtn.addEventListener('click', function () {
      applyPlaybackSpeed(state.speed - 0.05);
    });
  }

  const speedUpBtn = document.getElementById('speedUpBtn');
  if (speedUpBtn) {
    speedUpBtn.addEventListener('click', function () {
      applyPlaybackSpeed(state.speed + 0.05);
    });
  }

  // Metronome & Count-In Buttons
  const metronomeBtn = document.getElementById('metronomeBtn');
  if (metronomeBtn) {
    metronomeBtn.addEventListener('click', function () {
      state.metronomeOn = !state.metronomeOn;
      metronomeBtn.classList.toggle('active', state.metronomeOn);
    });
  }

  const countInBtn = document.getElementById('countInBtn');
  if (countInBtn) {
    countInBtn.addEventListener('click', function () {
      state.countInOn = !state.countInOn;
      countInBtn.classList.toggle('active', state.countInOn);
    });
  }

  // Wait Mode Switch
  document.getElementById('waitSwitch').addEventListener('click', function (e) {
    state.waitMode = !state.waitMode;
    e.currentTarget.classList.toggle('active-blue', state.waitMode);
    document.getElementById('waitOrb').classList.toggle('orb-dim', !state.waitMode);
    if (!state.waitMode) {
      state.waitingMidis.clear();
      state.waitingNoteIndices.clear();
      state.waitStartedAtMs = null;
      updateWaitBanner();
    }
  });

  function pruneWaitingNotesForActiveHands() {
    if (!state.waitMode || state.waitingNoteIndices.size === 0 || !state.song || !Array.isArray(state.song.notes)) {
      return;
    }
    const nextMidis = new Set();
    Array.from(state.waitingNoteIndices).forEach(function (idx) {
      const n = state.song.notes[idx];
      if (!n) {
        state.waitingNoteIndices.delete(idx);
        return;
      }
      const handActive = !state.handIsoMaster ||
        (n.hand === 'L' && state.leftHandOn) ||
        (n.hand === 'R' && state.rightHandOn);
      if (!handActive) {
        state.waitingNoteIndices.delete(idx);
        state.clearedWaitNoteIndices.add(idx);
        state.playedIndices.add(idx);
      } else {
        nextMidis.add(n.midi);
      }
    });
    state.waitingMidis = nextMidis;
    if (state.waitingMidis.size === 0) {
      state.waitStartedAtMs = null;
    }
    updateWaitBanner();
  }

  // Hand Isolation Controls
  function syncHandIsolationUI() {
    const sw = document.getElementById('handIsoSwitch');
    const orb = document.getElementById('handOrb');
    const lBtn = document.getElementById('leftHandBtn');
    const rBtn = document.getElementById('rightHandBtn');

    if (sw) {
      sw.classList.toggle('active-green', state.handIsoMaster);
      sw.setAttribute('aria-checked', state.handIsoMaster ? 'true' : 'false');
    }
    if (orb) {
      orb.classList.toggle('orb-dim', !state.handIsoMaster);
    }
    if (lBtn) {
      lBtn.className = 'hand-pill hand-pill-l ' + (state.handIsoMaster && state.leftHandOn ? 'active' : 'inactive');
    }
    if (rBtn) {
      rBtn.className = 'hand-pill hand-pill-r ' + (state.handIsoMaster && state.rightHandOn ? 'active' : 'inactive');
    }
    pruneWaitingNotesForActiveHands();
  }

  document.getElementById('handIsoSwitch').addEventListener('click', function () {
    state.handIsoMaster = !state.handIsoMaster;
    if (state.handIsoMaster) {
      const targetHand = state.lastIsolatedHand || 'R';
      state.leftHandOn = (targetHand === 'L');
      state.rightHandOn = (targetHand === 'R');
    } else {
      state.leftHandOn = false;
      state.rightHandOn = false;
    }
    syncHandIsolationUI();
  });

  document.getElementById('leftHandBtn').addEventListener('click', function () {
    if (!state.handIsoMaster || !state.leftHandOn) {
      state.handIsoMaster = true;
      state.leftHandOn = true;
      state.rightHandOn = false;
      state.lastIsolatedHand = 'L';
    } else {
      state.handIsoMaster = false;
      state.leftHandOn = false;
      state.rightHandOn = false;
    }
    syncHandIsolationUI();
  });

  document.getElementById('rightHandBtn').addEventListener('click', function () {
    if (!state.handIsoMaster || !state.rightHandOn) {
      state.handIsoMaster = true;
      state.rightHandOn = true;
      state.leftHandOn = false;
      state.lastIsolatedHand = 'R';
    } else {
      state.handIsoMaster = false;
      state.leftHandOn = false;
      state.rightHandOn = false;
    }
    syncHandIsolationUI();
  });

  // Sound Preset Selector & Felt Softness Slider
  document.getElementById('soundCollapseBtn').addEventListener('click', function () {
    const body = document.getElementById('soundSectionBody');
    const chev = document.getElementById('soundChevron');
    const hidden = body.style.display === 'none';
    body.style.display = hidden ? 'flex' : 'none';
    chev.style.transform = hidden ? 'rotate(0deg)' : 'rotate(-90deg)';
  });

  const PRESET_SHORT_LABELS = {
    acoustic_grand: 'Acoustic Grand',
    mellow_felt: 'Mellow Felt',
    warm_upright: 'Vintage Upright',
    electric_piano: 'Electric Piano'
  };

  const presetBtns = document.querySelectorAll('.sound-option-btn');
  presetBtns.forEach(function (btn) {
    btn.addEventListener('click', function () {
      const preset = btn.getAttribute('data-preset');
      audioEngine.setPreset(preset);

      document.getElementById('warmthSlider').value = audioEngine.warmth;
      document.getElementById('warmthLabel').textContent = audioEngine.warmth + '%';

      const activeNameEl = document.getElementById('soundActiveName');
      if (activeNameEl && PRESET_SHORT_LABELS[preset]) {
        activeNameEl.textContent = PRESET_SHORT_LABELS[preset];
      }

      presetBtns.forEach(function (b) {
        b.classList.toggle('selected', b === btn);
      });

      // Audition warm chord when selecting a piano sound
      [60, 64, 67, 72].forEach(function (m, i) {
        setTimeout(function () { audioEngine.playNote(m, 0.9, 0.72); }, i * 45);
      });
    });
  });

  document.getElementById('warmthSlider').addEventListener('input', function (e) {
    const val = parseInt(e.target.value, 10);
    document.getElementById('warmthLabel').textContent = val + '%';
    audioEngine.setPreset(audioEngine.preset, val);
  });

  // Settings Popover
  const settingsPopover = document.getElementById('settingsPopover');
  const settingsBtn = document.getElementById('settingsToggleBtn');
  settingsBtn.addEventListener('click', function (e) {
    e.stopPropagation();
    settingsPopover.classList.toggle('open');
  });
  window.addEventListener('click', function (e) {
    if (!settingsPopover.contains(e.target) && e.target !== settingsBtn) {
      settingsPopover.classList.remove('open');
    }
  });

  document.querySelectorAll('.horizon-opt').forEach(function (btn) {
    btn.addEventListener('click', function () {
      state.horizonSec = parseFloat(btn.getAttribute('data-sec'));
      document.getElementById('horizonLabel').textContent = Math.round(state.horizonSec) + 's';
      document.querySelectorAll('.horizon-opt').forEach(function (b) {
        b.classList.toggle('active', b === btn);
      });
    });
  });

  const chordDetectorSwitch = document.getElementById('chordDetectorSwitch');
  if (chordDetectorSwitch) {
    chordDetectorSwitch.addEventListener('click', function (e) {
      state.showChordDetector = !state.showChordDetector;
      e.currentTarget.classList.toggle('active-blue', state.showChordDetector);
      const chordBadge = document.getElementById('chordBadge');
      if (chordBadge && !state.showChordDetector) {
        chordBadge.style.display = 'none';
      }
    });
  }

  const measureGridSwitch = document.getElementById('measureGridSwitch');
  if (measureGridSwitch) {
    measureGridSwitch.addEventListener('click', function (e) {
      state.showMeasureGrid = !state.showMeasureGrid;
      e.currentTarget.classList.toggle('active-blue', state.showMeasureGrid);
    });
  }

  document.getElementById('allKeyLabelsSwitch').addEventListener('click', function (e) {
    state.showAllKeyNames = !state.showAllKeyNames;
    e.currentTarget.classList.toggle('active-blue', state.showAllKeyNames);
  });

  // Synced A-B Loop Speed Ramp Controls (Popover Switch + Bottom Bar Pill)
  const speedRampSwitch = document.getElementById('speedRampSwitch');
  const loopRampBtn = document.getElementById('loopRampBtn');

  function setSpeedRamp(enabled) {
    state.speedRamp = !!enabled;
    if (state.speedRamp) {
      if (!state.loopEnabled) {
        state.loopEnabled = true;
        state.loopPassCount = 1;
        state.passHadLongWaitStall = false;
        state.passWrongKeys = 0;
        state.passMisses = 0;
        state.waitStartedAtMs = null;
        const loopBtnEl = document.getElementById('abLoopBtn');
        const loopBoxEl = document.getElementById('abLoopBox');
        const hAEl = document.getElementById('loopHandleA');
        const hBEl = document.getElementById('loopHandleB');
        if (loopBtnEl) loopBtnEl.classList.add('active');
        if (loopBoxEl) loopBoxEl.classList.add('visible');
        if (hAEl) hAEl.classList.add('visible');
        if (hBEl) hBEl.classList.add('visible');
      }
      if (state.speed > 0.5) {
        applyPlaybackSpeed(0.5);
      }
    }
    if (speedRampSwitch) {
      speedRampSwitch.classList.toggle('active-green', state.speedRamp);
    }
    if (loopRampBtn) {
      loopRampBtn.classList.toggle('active', state.speedRamp);
    }
    syncLoopUI();
  }

  if (speedRampSwitch) {
    speedRampSwitch.addEventListener('click', function () {
      setSpeedRamp(!state.speedRamp);
    });
  }
  if (loopRampBtn) {
    loopRampBtn.addEventListener('click', function () {
      setSpeedRamp(!state.speedRamp);
    });
  }

  // Export Active Song as Standard .MID File (Format 1 with L/R Hand Tracks)
  const exportMidiBtn = document.getElementById('exportMidiBtn');
  if (exportMidiBtn) {
    exportMidiBtn.addEventListener('click', function () {
      downloadSongMidi(state.song);
    });
  }

  // Delete Custom Song or Reset Modified Built-In Song from Settings Popover
  const deleteCustomSongBtn = document.getElementById('deleteCustomSongBtn');
  if (deleteCustomSongBtn) {
    deleteCustomSongBtn.addEventListener('click', function () {
      deleteOrResetSong(state.activeSongId);
      settingsPopover.classList.remove('open');
    });
  }

  /********************************************************************
   * PHASE 3: AI AUDIO-TO-MIDI TRANSCRIBER MODAL ("BRING YOUR OWN SONG")
   ********************************************************************/
  const transcriberModal = document.getElementById('transcriberModal');
  const closeTranscriberModalBtn = document.getElementById('closeTranscriberModalBtn');
  const cancelTranscribeBtn = document.getElementById('cancelTranscribeBtn');
  const audioDropZone = document.getElementById('audioDropZone');
  const audioFileInput = document.getElementById('audioFileInput');
  const selectedAudioFileName = document.getElementById('selectedAudioFileName');
  const selectedAudioFileMeta = document.getElementById('selectedAudioFileMeta');
  const transcribeModeSelect = document.getElementById('transcribeModeSelect');
  const transcribeSensitivitySlider = document.getElementById('transcribeSensitivitySlider');
  const transcribeSensitivityLabel = document.getElementById('transcribeSensitivityLabel');
  const serverConfigBox = document.getElementById('serverConfigBox');
  const aiServerUrlInput = document.getElementById('aiServerUrlInput');
  const testAiServerBtn = document.getElementById('testAiServerBtn');
  const aiServerStatusPill = document.getElementById('aiServerStatusPill');
  const isolatePianoSwitch = document.getElementById('isolatePianoSwitch');
  const transcribeProgressWrap = document.getElementById('transcribeProgressWrap');
  const transcribeStatusText = document.getElementById('transcribeStatusText');
  const transcribePercentText = document.getElementById('transcribePercentText');
  const transcribeProgressFill = document.getElementById('transcribeProgressFill');
  const transcribeErrorBox = document.getElementById('transcribeErrorBox');
  const startTranscribeBtn = document.getElementById('startTranscribeBtn');

  let selectedAudioFile = null;
  let isolatePianoStem = true;
  let isTranscribing = false;

  /********************************************************************
   * AI SERVER URL PERSISTENCE
   ********************************************************************/
  function getSavedServerUrl() {
    try {
      const u = localStorage.getItem('keyflow_ai_server_url');
      if (u && typeof u === 'string' && u.trim().length > 0) {
        return u.trim();
      }
    } catch (e) {}
    return 'http://localhost:8000';
  }

  function setSavedServerUrl(url) {
    const clean = (url || '').trim() || 'http://localhost:8000';
    try {
      localStorage.setItem('keyflow_ai_server_url', clean);
    } catch (e) {}
    const aiInput = document.getElementById('aiServerUrlInput');
    if (aiInput && aiInput.value !== clean) aiInput.value = clean;
  }

  const initialServerUrl = getSavedServerUrl();
  if (aiServerUrlInput) aiServerUrlInput.value = initialServerUrl;

  function setTranscriberModalOpen(open) {
    if (!transcriberModal) return;
    transcriberModal.classList.toggle('open', !!open);
    transcriberModal.setAttribute('aria-hidden', open ? 'false' : 'true');
    if (open) {
      setPlaying(false);
      if (audioFileInput) audioFileInput.value = '';
      const currentUrl = getSavedServerUrl();
      if (aiServerUrlInput) aiServerUrlInput.value = currentUrl;
      refreshServerHealthStatus();
    }
  }

  function updateSelectedAudioFile(file) {
    selectedAudioFile = file || null;
    if (transcribeErrorBox) {
      transcribeErrorBox.style.display = 'none';
      transcribeErrorBox.textContent = '';
    }
    if (selectedAudioFile) {
      if (audioDropZone) audioDropZone.classList.add('has-file');
      const sizeMb = (selectedAudioFile.size / (1024 * 1024)).toFixed(1);
      if (selectedAudioFileName) {
        selectedAudioFileName.textContent = 'Selected: ' + selectedAudioFile.name + ' (' + sizeMb + ' MB)';
      }
      if (selectedAudioFileMeta) {
        selectedAudioFileMeta.textContent = 'Ready to convert into Left / Right hand piano notes';
      }
      if (startTranscribeBtn && !isTranscribing) {
        startTranscribeBtn.disabled = false;
      }
    } else {
      if (audioDropZone) audioDropZone.classList.remove('has-file');
      if (selectedAudioFileName) {
        selectedAudioFileName.textContent = 'Tap to choose an Audio File (.MP3, .WAV, .M4A, .OGG)';
      }
      if (selectedAudioFileMeta) {
        selectedAudioFileMeta.textContent = 'Automatically transcribes chords & melody into Left / Right hand falling notes';
      }
      if (startTranscribeBtn) {
        startTranscribeBtn.disabled = true;
      }
    }
  }

  let healthCheckReqId = 0;
  async function refreshServerHealthStatus() {
    if (!audioTranscriber || !aiServerUrlInput || !aiServerStatusPill) return;
    const reqId = ++healthCheckReqId;
    const url = aiServerUrlInput.value.trim() || getSavedServerUrl();
    setSavedServerUrl(url);
    aiServerStatusPill.className = 'server-status-pill';
    aiServerStatusPill.textContent = 'Checking...';
    const status = await audioTranscriber.checkServerHealth(url, 4000);
    if (reqId !== healthCheckReqId) return;
    if (status.ok) {
      aiServerStatusPill.className = 'server-status-pill online';
      aiServerStatusPill.textContent = 'Online';
    } else {
      aiServerStatusPill.className = 'server-status-pill offline';
      aiServerStatusPill.textContent = 'Offline';
    }
  }

  if (aiServerUrlInput) {
    aiServerUrlInput.addEventListener('input', function () {
      setSavedServerUrl(aiServerUrlInput.value);
    });
  }

  // Phase 4B: Song Library Backup (.keyflow.json) & Restore
  const exportLibraryBtn = document.getElementById('exportLibraryBtn');
  if (exportLibraryBtn) {
    exportLibraryBtn.addEventListener('click', async function () {
      try {
        if (!audioTranscriber || !audioTranscriber.customSongStore || !audioTranscriber.customSongStore.exportLibraryJson) {
          alert('Custom song store is not available.');
          return;
        }
        await audioTranscriber.customSongStore.exportLibraryJson();
      } catch (err) {
        alert('Failed to export library: ' + (err && err.message ? err.message : err));
      }
    });
  }

  const importLibraryInput = document.getElementById('importLibraryInput');
  if (importLibraryInput) {
    importLibraryInput.addEventListener('change', async function (e) {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      try {
        const text = await file.text();
        if (!audioTranscriber || !audioTranscriber.customSongStore || !audioTranscriber.customSongStore.importLibraryJson) {
          throw new Error('Custom song store is not available');
        }
        const result = await audioTranscriber.customSongStore.importLibraryJson(text);
        if (result && Array.isArray(result.songs)) {
          result.songs.forEach(function (s) {
            if (!s || !s.id || s.id === '__proto__' || s.id === 'constructor' || s.id === 'prototype') return;
            SONGS[s.id] = s;
          });
        }
        renderHomeSongGrid();
        alert('Successfully restored ' + (result ? result.count : 0) + ' song(s) to your library!');
      } catch (err) {
        alert('Failed to restore library: ' + (err && err.message ? err.message : err));
      } finally {
        importLibraryInput.value = '';
      }
    });
  }

  function setAddSongModalOpen(open) {
    if (!addSongModal) return;
    addSongModal.classList.toggle('open', !!open);
    addSongModal.setAttribute('aria-hidden', open ? 'false' : 'true');
  }

  if (homeAddSongBtn) {
    homeAddSongBtn.addEventListener('click', function () {
      setAddSongModalOpen(true);
    });
  }
  if (closeAddSongModalBtn) {
    closeAddSongModalBtn.addEventListener('click', function () {
      setAddSongModalOpen(false);
    });
  }
  if (addSongModal) {
    addSongModal.addEventListener('click', function (e) {
      if (e.target === addSongModal) {
        setAddSongModalOpen(false);
      }
    });
  }
  if (homeOpenMidiBtn) {
    homeOpenMidiBtn.addEventListener('click', function () {
      setAddSongModalOpen(false);
    });
  }

  if (homeOpenAiModalBtn) {
    homeOpenAiModalBtn.addEventListener('click', function () {
      setAddSongModalOpen(false);
      setTranscriberModalOpen(true);
    });
  }
  if (closeTranscriberModalBtn) {
    closeTranscriberModalBtn.addEventListener('click', function () {
      if (!isTranscribing) setTranscriberModalOpen(false);
    });
  }
  if (cancelTranscribeBtn) {
    cancelTranscribeBtn.addEventListener('click', function () {
      if (!isTranscribing) setTranscriberModalOpen(false);
    });
  }
  if (transcriberModal) {
    transcriberModal.addEventListener('click', function (e) {
      if (e.target === transcriberModal && !isTranscribing) {
        setTranscriberModalOpen(false);
      }
    });
  }

  if (audioFileInput) {
    audioFileInput.addEventListener('change', function (e) {
      const f = e.target.files && e.target.files[0];
      if (f) updateSelectedAudioFile(f);
    });
  }

  if (audioDropZone) {
    audioDropZone.addEventListener('click', function (e) {
      if (isTranscribing) {
        e.preventDefault();
      }
    });
    audioDropZone.addEventListener('dragover', function (e) {
      e.preventDefault();
      audioDropZone.classList.add('dragover');
    });
    audioDropZone.addEventListener('dragleave', function () {
      audioDropZone.classList.remove('dragover');
    });
    audioDropZone.addEventListener('drop', function (e) {
      e.preventDefault();
      audioDropZone.classList.remove('dragover');
      if (isTranscribing) return;
      const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) updateSelectedAudioFile(f);
    });
  }

  // Home Screen Overflow Menu (Backup / Restore)
  const homeMoreMenuBtn = document.getElementById('homeMoreMenuBtn');
  const homeMoreMenu = document.getElementById('homeMoreMenu');
  if (homeMoreMenuBtn && homeMoreMenu) {
    const closeHomeMoreMenu = function () {
      homeMoreMenu.classList.remove('open');
      homeMoreMenuBtn.setAttribute('aria-expanded', 'false');
    };
    homeMoreMenuBtn.addEventListener('click', function (e) {
      e.stopPropagation();
      const willOpen = !homeMoreMenu.classList.contains('open');
      homeMoreMenu.classList.toggle('open', willOpen);
      homeMoreMenuBtn.setAttribute('aria-expanded', willOpen ? 'true' : 'false');
    });
    homeMoreMenu.addEventListener('click', function () {
      closeHomeMoreMenu();
    });
    document.addEventListener('click', function (e) {
      if (!homeMoreMenu.contains(e.target) && e.target !== homeMoreMenuBtn) {
        closeHomeMoreMenu();
      }
    });
  }

  // Home Screen Drag-and-Drop (.MID or Audio files)
  const homeViewEl = document.getElementById('homeView');
  const homeDropOverlay = document.getElementById('homeDropOverlay');
  if (homeViewEl && homeDropOverlay) {
    let homeDragDepth = 0;
    const isFileDrag = function (e) {
      const types = e.dataTransfer && e.dataTransfer.types;
      if (!types) return false;
      for (let i = 0; i < types.length; i++) {
        if (types[i] === 'Files') return true;
      }
      return false;
    };
    const setDropOverlayVisible = function (show) {
      homeDropOverlay.classList.toggle('visible', !!show);
      homeDropOverlay.setAttribute('aria-hidden', show ? 'false' : 'true');
    };

    homeViewEl.addEventListener('dragenter', function (e) {
      if (!isFileDrag(e)) return;
      e.preventDefault();
      homeDragDepth++;
      setDropOverlayVisible(true);
    });
    homeViewEl.addEventListener('dragover', function (e) {
      if (!isFileDrag(e)) return;
      e.preventDefault();
    });
    homeViewEl.addEventListener('dragleave', function (e) {
      if (!isFileDrag(e)) return;
      homeDragDepth = Math.max(0, homeDragDepth - 1);
      if (homeDragDepth === 0) setDropOverlayVisible(false);
    });
    homeViewEl.addEventListener('drop', function (e) {
      if (!isFileDrag(e)) return;
      e.preventDefault();
      homeDragDepth = 0;
      setDropOverlayVisible(false);
      const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (!f) return;
      const lower = String(f.name || '').toLowerCase();
      if (lower.endsWith('.mid') || lower.endsWith('.midi')) {
        importMidiFile(f);
      } else if (
        (f.type && f.type.indexOf('audio/') === 0) ||
        /\.(mp3|wav|m4a|aac|ogg|flac)$/i.test(lower)
      ) {
        setTranscriberModalOpen(true);
        updateSelectedAudioFile(f);
      }
    });
  }

  // Editor Batch Tools Popover (Step 3)
  const editorToolsToggleBtn = document.getElementById('editorToolsToggleBtn');
  const editorToolsPopover = document.getElementById('editorToolsPopover');
  if (editorToolsToggleBtn && editorToolsPopover) {
    const closeEditorToolsPopover = function () {
      editorToolsPopover.classList.remove('open');
      editorToolsToggleBtn.classList.remove('active');
      editorToolsToggleBtn.setAttribute('aria-expanded', 'false');
    };
    editorToolsToggleBtn.addEventListener('click', function (e) {
      e.stopPropagation();
      const willOpen = !editorToolsPopover.classList.contains('open');
      editorToolsPopover.classList.toggle('open', willOpen);
      editorToolsToggleBtn.classList.toggle('active', willOpen);
      editorToolsToggleBtn.setAttribute('aria-expanded', willOpen ? 'true' : 'false');
    });
    editorToolsPopover.addEventListener('click', function (e) {
      e.stopPropagation();
    });
    document.addEventListener('click', function (e) {
      if (!editorToolsPopover.contains(e.target) && e.target !== editorToolsToggleBtn) {
        closeEditorToolsPopover();
      }
    });
  }

  // Collapsible Advanced AI Server Config (Step 4)
  const advancedServerWrap = document.getElementById('advancedServerWrap');
  const advancedServerToggleBtn = document.getElementById('advancedServerToggleBtn');
  let advancedServerExpanded = false;

  function syncAdvancedServerUI() {
    if (!serverConfigBox) return;
    const mode = transcribeModeSelect ? transcribeModeSelect.value : 'auto';
    if (mode === 'browser') {
      if (advancedServerWrap) advancedServerWrap.style.display = 'none';
      serverConfigBox.style.display = 'none';
      return;
    }
    if (advancedServerWrap) advancedServerWrap.style.display = 'flex';
    const shouldShowBox = (mode === 'server') || advancedServerExpanded;
    serverConfigBox.style.display = shouldShowBox ? 'flex' : 'none';
    if (advancedServerToggleBtn) {
      advancedServerToggleBtn.classList.toggle('open', shouldShowBox);
      advancedServerToggleBtn.setAttribute('aria-expanded', shouldShowBox ? 'true' : 'false');
    }
  }

  if (advancedServerToggleBtn) {
    advancedServerToggleBtn.addEventListener('click', function () {
      advancedServerExpanded = !advancedServerExpanded;
      syncAdvancedServerUI();
    });
  }

  if (transcribeModeSelect && serverConfigBox) {
    transcribeModeSelect.addEventListener('change', function () {
      if (transcribeModeSelect.value === 'server') {
        advancedServerExpanded = true;
      }
      syncAdvancedServerUI();
    });
    syncAdvancedServerUI();
  }

  if (transcribeSensitivitySlider && transcribeSensitivityLabel) {
    transcribeSensitivitySlider.addEventListener('input', function () {
      transcribeSensitivityLabel.textContent = transcribeSensitivitySlider.value + '%';
    });
  }

  if (isolatePianoSwitch) {
    isolatePianoSwitch.addEventListener('click', function () {
      isolatePianoStem = !isolatePianoStem;
      isolatePianoSwitch.classList.toggle('active-green', isolatePianoStem);
      isolatePianoSwitch.setAttribute('aria-checked', isolatePianoStem ? 'true' : 'false');
    });
  }

  if (testAiServerBtn) {
    testAiServerBtn.addEventListener('click', function () {
      refreshServerHealthStatus();
    });
  }

  if (startTranscribeBtn) {
    startTranscribeBtn.addEventListener('click', async function () {
      if (!selectedAudioFile || isTranscribing || !audioTranscriber) return;
      isTranscribing = true;
      startTranscribeBtn.disabled = true;
      if (transcribeErrorBox) transcribeErrorBox.style.display = 'none';
      if (transcribeProgressWrap) transcribeProgressWrap.style.display = 'flex';

      const engineMode = transcribeModeSelect ? transcribeModeSelect.value : 'auto';
      const sensitivity = transcribeSensitivitySlider
        ? (parseInt(transcribeSensitivitySlider.value, 10) || 55)
        : 55;
      const serverUrl = (aiServerUrlInput ? aiServerUrlInput.value.trim() : '') || getSavedServerUrl();
      setSavedServerUrl(serverUrl);

      try {
        const songObj = await audioTranscriber.transcribeAudioFile(
          selectedAudioFile,
          {
            mode: engineMode,
            engineMode: engineMode,
            serverUrl: serverUrl,
            isolatePiano: isolatePianoStem,
            sensitivity: sensitivity
          },
          function (prog) {
            if (transcribeStatusText && prog.stage) transcribeStatusText.textContent = prog.stage;
            if (typeof prog.percent === 'number') {
              const p = Math.max(0, Math.min(100, Math.round(prog.percent)));
              if (transcribePercentText) transcribePercentText.textContent = p + '%';
              if (transcribeProgressFill) transcribeProgressFill.style.width = p + '%';
            }
          }
        );

        const id = 'ai_' + Date.now();
        songObj.id = id;
        songObj.isCustom = true;
        songObj.createdAt = Date.now();
        SONGS[id] = songObj;

        if (audioTranscriber.customSongStore) {
          await audioTranscriber.customSongStore.saveSong(songObj);
        }

        isTranscribing = false;
        if (transcribeProgressWrap) transcribeProgressWrap.style.display = 'none';
        startTranscribeBtn.disabled = false;
        setTranscriberModalOpen(false);
        renderHomeSongGrid();
        loadSongById(id);
        setViewMode('play');
      } catch (err) {
        isTranscribing = false;
        startTranscribeBtn.disabled = false;
        if (transcribeErrorBox) {
          transcribeErrorBox.style.display = 'block';
          transcribeErrorBox.textContent = (err && err.message) ? err.message : 'Could not transcribe audio file.';
        }
      }
    });
  }

  // Audio Mute Button
  document.getElementById('muteBtn').addEventListener('click', function () {
    if (typeof audioEngine.setMuted === 'function') {
      audioEngine.setMuted(!audioEngine.muted);
    } else {
      audioEngine.muted = !audioEngine.muted;
    }
    document.getElementById('volOnSvg').style.display = audioEngine.muted ? 'none' : 'block';
    document.getElementById('volOffSvg').style.display = audioEngine.muted ? 'block' : 'none';
  });

  // A-B Loop & Interactive Scrubbing
  const abLoopBtn = document.getElementById('abLoopBtn');
  const abLoopBox = document.getElementById('abLoopBox');
  const handleA = document.getElementById('loopHandleA');
  const handleB = document.getElementById('loopHandleB');
  const timelineTrack = document.getElementById('timelineTrack');

  abLoopBtn.addEventListener('click', function () {
    state.loopEnabled = !state.loopEnabled;
    state.loopPassCount = 1;
    state.passHadLongWaitStall = false;
    state.passWrongKeys = 0;
    state.passMisses = 0;
    state.waitStartedAtMs = null;
    abLoopBtn.classList.toggle('active', state.loopEnabled);
    abLoopBox.classList.toggle('visible', state.loopEnabled);
    handleA.classList.toggle('visible', state.loopEnabled);
    handleB.classList.toggle('visible', state.loopEnabled);
    syncLoopUI();
  });

  let dragTarget = null;
  function eventToSongSec(e) {
    const rect = timelineTrack.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left) / Math.max(1, rect.width)));
    return ratio * state.song.duration;
  }

  handleA.addEventListener('pointerdown', function (e) {
    e.stopPropagation();
    dragTarget = 'A';
    state.loopPassCount = 1;
    state.passHadLongWaitStall = false;
  });
  handleB.addEventListener('pointerdown', function (e) {
    e.stopPropagation();
    dragTarget = 'B';
    state.loopPassCount = 1;
    state.passHadLongWaitStall = false;
  });
  timelineTrack.addEventListener('pointerdown', function (e) {
    if (e.target === handleA || e.target === handleB) return;
    dragTarget = 'playhead';
    seekTo(eventToSongSec(e));
  });

  window.addEventListener('pointermove', function (e) {
    if (!dragTarget) return;
    const sec = eventToSongSec(e);
    if (dragTarget === 'playhead') {
      seekTo(sec);
    } else if (dragTarget === 'A') {
      state.loopA = Math.max(0, Math.min(state.loopB - 1.5, sec));
      syncLoopUI();
    } else if (dragTarget === 'B') {
      state.loopB = Math.min(state.song.duration, Math.max(state.loopA + 1.5, sec));
      syncLoopUI();
    }
  });

  window.addEventListener('pointerup', function () {
    dragTarget = null;
  });

  // Responsive Canvas Resize Listeners (with visualViewport & ResizeObserver for iPad Safari/Chrome)
  function handleViewportResize() {
    syncViewportHeight();
    rebuildKeyboardGeometry();
  }

  window.addEventListener('resize', handleViewportResize);
  window.addEventListener('orientationchange', function () {
    setTimeout(handleViewportResize, 80);
  });
  try {
    if (window.visualViewport) {
      window.visualViewport.addEventListener('resize', handleViewportResize);
    }
  } catch (e) {}
  try {
    if (window.ResizeObserver && stageEl) {
      const ro = new ResizeObserver(function () {
        rebuildKeyboardGeometry();
      });
      ro.observe(stageEl);
    }
  } catch (e) {}

  // Startup Initialization
  syncViewportHeight();
  syncHandIsolationUI();
  loadSongById('let_it_be');
  setViewMode('home');
  if (audioTranscriber && audioTranscriber.customSongStore) {
    audioTranscriber.customSongStore.getAllSongs().then(function (savedList) {
      if (Array.isArray(savedList)) {
        savedList.forEach(function (item) {
          if (item && item.id && item.id !== '__proto__' && item.id !== 'constructor' && item.id !== 'prototype' && Array.isArray(item.notes) && item.notes.length > 0) {
            if (BUILTIN_IDS.indexOf(item.id) !== -1) {
              item.isCustom = false;
              item.isModifiedBuiltin = true;
              item.source = 'builtin';
            } else {
              item.isCustom = true;
            }
            SONGS[item.id] = item;
          }
        });
        if (SONGS[state.activeSongId]) {
          loadSongById(state.activeSongId);
        }
        renderHomeSongGrid();
      }
    }).catch(function () {});
  }
  setTimeout(handleViewportResize, 120);
  requestAnimationFrame(tickLoop);
})(window);
