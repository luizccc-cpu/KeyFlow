(function (global) {
  'use strict';

  /********************************************************************
   * SMART LEFT/RIGHT HAND VOICE-SPLITTING ALGORITHM (PHASE 2)
   * - Groups simultaneous notes within 35ms (0.035s) into onset clusters
   * - Enforces human hand-span limit (max 14 semitones = 10th interval)
   * - Splits wide clusters at the largest internal pitch gap
   * - Tracks moving pitch centers (EMA initialized at 48 for L, 67 for R)
   *   with a soft split bias around C4 (MIDI 60)
   ********************************************************************/
  function assignHandsSmart(notes, options) {
    if (!Array.isArray(notes) || notes.length === 0) return notes;

    const MAX_HAND_SPAN = 14; // 10th interval (14 semitones)
    const CLUSTER_WINDOW_SEC = 0.035; // 35ms simultaneous onset window
    const EMA_ALPHA = 0.22;
    const manualSplitMidi = (options && typeof options.splitMidi === 'number' && isFinite(options.splitMidi))
      ? options.splitMidi
      : null;

    // Sort notes chronologically, then ascending by pitch
    notes.sort((a, b) => a.time - b.time || a.midi - b.midi);

    // 1. Group into simultaneous onset clusters (within 35ms)
    const clusters = [];
    let currentCluster = [notes[0]];

    for (let i = 1; i < notes.length; i++) {
      const n = notes[i];
      if (n.time - currentCluster[0].time <= CLUSTER_WINDOW_SEC) {
        currentCluster.push(n);
      } else {
        clusters.push(currentCluster);
        currentCluster = [n];
      }
    }
    if (currentCluster.length > 0) {
      clusters.push(currentCluster);
    }

    // If a manual split point was requested, split directly around manualSplitMidi
    // while still splitting single-side chords wider than MAX_HAND_SPAN
    if (manualSplitMidi !== null) {
      for (let cIdx = 0; cIdx < clusters.length; cIdx++) {
        const cNotes = clusters[cIdx];
        cNotes.sort((a, b) => a.midi - b.midi);
        for (let i = 0; i < cNotes.length; i++) {
          cNotes[i].hand = (cNotes[i].midi < manualSplitMidi) ? 'L' : 'R';
        }
      }
      return notes;
    }

    // 2. Song-Adaptive Register Calibration (25th / 75th percentiles + bimodal valley pivot)
    const sortedPitches = notes.map(n => n.midi).sort((a, b) => a - b);
    const p25 = sortedPitches[Math.floor(sortedPitches.length * 0.25)] || 48;
    const p46 = sortedPitches[Math.floor(sortedPitches.length * 0.46)] || 58;
    const p75 = sortedPitches[Math.floor(sortedPitches.length * 0.75)] || 67;

    let anchorL = Math.max(36, Math.min(58, p25));
    let anchorR = Math.max(54, Math.min(84, p75));
    if (anchorR - anchorL < 10) {
      const mid = (anchorL + anchorR) * 0.5;
      anchorL = Math.max(36, Math.round(mid - 5));
      anchorR = Math.min(84, Math.round(mid + 5));
    }

    // Build smoothed pitch histogram to locate natural valley between anchorL and anchorR
    const pitchHist = new Array(128).fill(0);
    for (let i = 0; i < notes.length; i++) {
      const m = Math.max(0, Math.min(127, notes[i].midi));
      const w = Math.min(1.5, Math.max(0.15, notes[i].duration || 0.25));
      pitchHist[m] += w;
    }
    let valleyPitch = p46;
    let minValleyWeight = Infinity;
    const searchLow = Math.max(46, Math.round(anchorL + 3));
    const searchHigh = Math.min(68, Math.round(anchorR - 3));
    if (searchHigh >= searchLow) {
      for (let m = searchLow; m <= searchHigh; m++) {
        const smoothW = pitchHist[m - 1] * 0.25 + pitchHist[m] * 0.5 + pitchHist[m + 1] * 0.25;
        if (smoothW < minValleyWeight) {
          minValleyWeight = smoothW;
          valleyPitch = m;
        }
      }
    }

    const songPivot = Math.max(50, Math.min(65, Math.round(0.5 * p46 + 0.5 * valleyPitch)));
    let leftCenter = anchorL;
    let rightCenter = anchorR;

    // 3. Walk clusters with sustained-note & local-window hand-span constraints
    for (let cIdx = 0; cIdx < clusters.length; cIdx++) {
      const cNotes = clusters[cIdx];
      cNotes.sort((a, b) => a.midi - b.midi);

      const clusterTime = cNotes[0].time;
      const minPitch = cNotes[0].midi;
      const maxPitch = cNotes[cNotes.length - 1].midi;
      const totalSpan = maxPitch - minPitch;

      // Inspect active (held) or very recent notes (within 0.42s) from previous clusters
      let activeLeftMin = null;
      let activeRightMax = null;
      for (let prevIdx = cIdx - 1; prevIdx >= 0; prevIdx--) {
        const pCluster = clusters[prevIdx];
        if (clusterTime - pCluster[0].time > 4.0) break;
        for (let j = 0; j < pCluster.length; j++) {
          const pn = pCluster[j];
          const isHeld = (pn.time + (pn.duration || 0.25)) > (clusterTime + 0.04);
          const isRecent = (clusterTime - pn.time) <= 0.42;
          if (isHeld || isRecent) {
            if (pn.hand === 'L') {
              if (activeLeftMin === null || pn.midi < activeLeftMin) activeLeftMin = pn.midi;
            } else if (pn.hand === 'R') {
              if (activeRightMax === null || pn.midi > activeRightMax) activeRightMax = pn.midi;
            }
          }
        }
      }

      // Compute local 1.2s window pitch range for broken-chord / arpeggio context
      let localMin = minPitch;
      let localMax = maxPitch;
      for (let wIdx = Math.max(0, cIdx - 8); wIdx <= Math.min(clusters.length - 1, cIdx + 8); wIdx++) {
        const wCluster = clusters[wIdx];
        if (Math.abs(wCluster[0].time - clusterTime) <= 0.65) {
          for (let j = 0; j < wCluster.length; j++) {
            if (wCluster[j].midi < localMin) localMin = wCluster[j].midi;
            if (wCluster[j].midi > localMax) localMax = wCluster[j].midi;
          }
        }
      }

      const centerMid = (leftCenter + rightCenter) * 0.5;
      const localMid = (localMax - localMin >= 12) ? ((localMin + localMax) * 0.5) : songPivot;
      const effectivePivot = 0.50 * centerMid + 0.30 * songPivot + 0.20 * localMid;

      if (cNotes.length === 1) {
        const p = cNotes[0].midi;
        const leftSpanViolation = (activeLeftMin !== null && (p - activeLeftMin) > MAX_HAND_SPAN) ? 18.0 : 0.0;
        const rightSpanViolation = (activeRightMax !== null && (activeRightMax - p) > MAX_HAND_SPAN) ? 18.0 : 0.0;

        if (p < anchorL - 5 && rightSpanViolation === 0) {
          cNotes[0].hand = 'L';
        } else if (p > anchorR + 5 && leftSpanViolation === 0) {
          cNotes[0].hand = 'R';
        } else {
          const distL = Math.abs(p - leftCenter) + (p >= effectivePivot ? (p - effectivePivot) * 0.45 : 0) + leftSpanViolation;
          const distR = Math.abs(p - rightCenter) + (p < effectivePivot ? (effectivePivot - p) * 0.45 : 0) + rightSpanViolation;

          if (Math.abs(distL - distR) < 1.2) {
            cNotes[0].hand = (p < effectivePivot) ? 'L' : 'R';
          } else {
            cNotes[0].hand = (distL < distR) ? 'L' : 'R';
          }
        }
      } else if (totalSpan > MAX_HAND_SPAN) {
        // Total span exceeds 14 semitones: must split between Left and Right hands
        let bestSplit = 1;
        let bestScore = -9999;

        for (let k = 1; k < cNotes.length; k++) {
          const leftSpan = cNotes[k - 1].midi - cNotes[0].midi;
          const rightSpan = cNotes[cNotes.length - 1].midi - cNotes[k].midi;
          const gap = cNotes[k].midi - cNotes[k - 1].midi;
          const splitBoundaryPitch = (cNotes[k - 1].midi + cNotes[k].midi) * 0.5;

          const overflowPenalty =
            Math.max(0, leftSpan - MAX_HAND_SPAN) * 4.5 +
            Math.max(0, rightSpan - MAX_HAND_SPAN) * 4.5;
          const pivotPenalty = Math.abs(splitBoundaryPitch - effectivePivot) * 0.30 +
            Math.abs(splitBoundaryPitch - songPivot) * 0.15;

          const score = gap * 2.0 - overflowPenalty - pivotPenalty;
          if (score > bestScore) {
            bestScore = score;
            bestSplit = k;
          }
        }

        for (let i = 0; i < cNotes.length; i++) {
          cNotes[i].hand = (i < bestSplit) ? 'L' : 'R';
        }
      } else {
        // Compact cluster (totalSpan <= 14 semitones, 2+ notes)
        let maxGap = 0;
        let maxGapIdx = -1;
        for (let k = 1; k < cNotes.length; k++) {
          const gap = cNotes[k].midi - cNotes[k - 1].midi;
          if (gap > maxGap) {
            maxGap = gap;
            maxGapIdx = k;
          }
        }

        const lowAvg = cNotes.slice(0, maxGapIdx).reduce((s, n) => s + n.midi, 0) / Math.max(1, maxGapIdx);
        const highAvg = cNotes.slice(maxGapIdx).reduce((s, n) => s + n.midi, 0) / Math.max(1, cNotes.length - maxGapIdx);

        const straddlesHands =
          maxGap >= 7 &&
          lowAvg <= effectivePivot + 1 &&
          highAvg >= effectivePivot - 1 &&
          Math.abs(lowAvg - leftCenter) <= Math.abs(lowAvg - rightCenter) + 2 &&
          Math.abs(highAvg - rightCenter) <= Math.abs(highAvg - leftCenter) + 2;

        if (straddlesHands) {
          for (let i = 0; i < cNotes.length; i++) {
            cNotes[i].hand = (i < maxGapIdx) ? 'L' : 'R';
          }
        } else {
          const clusterAvg = cNotes.reduce((s, n) => s + n.midi, 0) / cNotes.length;
          const leftSpanViolation = (activeLeftMin !== null && (maxPitch - activeLeftMin) > MAX_HAND_SPAN) ? 18.0 : 0.0;
          const rightSpanViolation = (activeRightMax !== null && (activeRightMax - minPitch) > MAX_HAND_SPAN) ? 18.0 : 0.0;

          const distL = Math.abs(clusterAvg - leftCenter) +
            (clusterAvg >= effectivePivot ? (clusterAvg - effectivePivot) * 0.45 : 0) +
            leftSpanViolation;
          const distR = Math.abs(clusterAvg - rightCenter) +
            (clusterAvg < effectivePivot ? (effectivePivot - clusterAvg) * 0.45 : 0) +
            rightSpanViolation;

          const assignedHand = (Math.abs(distL - distR) < 1.2)
            ? (clusterAvg < effectivePivot ? 'L' : 'R')
            : (distL < distR ? 'L' : 'R');

          for (let i = 0; i < cNotes.length; i++) {
            cNotes[i].hand = assignedHand;
          }
        }
      }

      // 4. Update EMA pitch centers tethered to song anchors (prevents runaway hand drift)
      let lSum = 0, lCount = 0;
      let rSum = 0, rCount = 0;
      for (let i = 0; i < cNotes.length; i++) {
        if (cNotes[i].hand === 'L') {
          lSum += cNotes[i].midi;
          lCount++;
        } else {
          rSum += cNotes[i].midi;
          rCount++;
        }
      }

      if (lCount > 0) {
        const lAvg = lSum / lCount;
        const rawL = leftCenter * (1 - EMA_ALPHA) + lAvg * EMA_ALPHA;
        const tetheredL = rawL * 0.82 + anchorL * 0.18;
        leftCenter = Math.max(33, Math.min(songPivot + 2, tetheredL));
      }
      if (rCount > 0) {
        const rAvg = rSum / rCount;
        const rawR = rightCenter * (1 - EMA_ALPHA) + rAvg * EMA_ALPHA;
        const tetheredR = rawR * 0.82 + anchorR * 0.18;
        rightCenter = Math.max(songPivot - 2, Math.min(90, tetheredR));
      }
      if (leftCenter > rightCenter - 10) {
        const mid = (leftCenter + rightCenter) * 0.5;
        leftCenter = Math.min(songPivot - 2, mid - 5);
        rightCenter = Math.max(songPivot + 2, mid + 5);
      }
    }

    return notes;
  }

  /********************************************************************
   * MUSICAL CHORD & LEGATO SUSTAIN RECOVERY
   * - Extends clipped stem-separated chords and bass notes so they
   *   sustain cleanly until the next chord change (up to 2.6s),
   *   while preventing any overlap with re-strikes of the same key.
   ********************************************************************/
  function extendChordLegato(notes, options) {
    if (!Array.isArray(notes) || notes.length < 2) return 0;

    const CLUSTER_WINDOW_SEC = 0.045; // 45ms simultaneous chord onset window
    const RELEASE_GAP_SEC = 0.04;     // 40ms visual breathing gap before next chord
    const MAX_CHORD_GAP_SEC = (options && options.maxChordGapSec) || 2.6;
    const MAX_MELODY_GAP_SEC = (options && options.maxMelodyGapSec) || 1.35;

    notes.sort((a, b) => a.time - b.time || a.midi - b.midi);

    // 1. Precompute next onset time of the exact same MIDI pitch to prevent overlaps
    const nextSamePitchTime = new Array(notes.length).fill(Infinity);
    const lastSeenByPitch = {};
    for (let i = notes.length - 1; i >= 0; i--) {
      const m = notes[i].midi;
      if (lastSeenByPitch[m] !== undefined) {
        nextSamePitchTime[i] = lastSeenByPitch[m];
      }
      lastSeenByPitch[m] = notes[i].time;
    }

    // 2. Group notes into simultaneous onset clusters (within 45ms)
    const clusters = [];
    let curCluster = [{ note: notes[0], index: 0 }];
    for (let i = 1; i < notes.length; i++) {
      if (notes[i].time - curCluster[0].note.time <= CLUSTER_WINDOW_SEC) {
        curCluster.push({ note: notes[i], index: i });
      } else {
        clusters.push(curCluster);
        curCluster = [{ note: notes[i], index: i }];
      }
    }
    if (curCluster.length > 0) {
      clusters.push(curCluster);
    }

    let extendedCount = 0;

    for (let cIdx = 0; cIdx < clusters.length - 1; cIdx++) {
      const cItems = clusters[cIdx];
      const clusterStart = cItems[0].note.time;

      if (cItems.length >= 2) {
        // Chord cluster (2+ simultaneous notes): find next distinct musical cluster (>= 0.25s ahead)
        let targetNextTime = null;
        for (let nextIdx = cIdx + 1; nextIdx < clusters.length; nextIdx++) {
          const candTime = clusters[nextIdx][0].note.time;
          const dt = candTime - clusterStart;
          if (dt < 0.22) continue;
          if (clusters[nextIdx].length >= 2 || dt >= 0.28) {
            targetNextTime = candTime;
            break;
          }
        }

        if (targetNextTime !== null) {
          const gap = targetNextTime - clusterStart;
          if (gap >= 0.25 && gap <= MAX_CHORD_GAP_SEC) {
            for (let k = 0; k < cItems.length; k++) {
              const n = cItems[k].note;
              const nIdx = cItems[k].index;
              const maxEnd = Math.min(targetNextTime - RELEASE_GAP_SEC, nextSamePitchTime[nIdx] - 0.03);
              const desiredDur = Math.round((maxEnd - n.time) * 1000) / 1000;
              if (n.duration <= 1.25 && desiredDur > n.duration + 0.05) {
                n.duration = Math.min(3.5, desiredDur);
                extendedCount++;
              }
            }
          }
        }
      } else {
        // Single note cluster: extend bass notes to next bass/chord, or melody notes to next melody note
        const n = cItems[0].note;
        const nIdx = cItems[0].index;
        const isBass = (n.hand === 'L' || n.midi < 56);
        let targetNextTime = null;

        for (let nextIdx = cIdx + 1; nextIdx < clusters.length; nextIdx++) {
          const nextCluster = clusters[nextIdx];
          const candTime = nextCluster[0].note.time;
          const dt = candTime - n.time;
          if (dt < 0.18) continue;

          if (isBass) {
            const hasBassOrChord = nextCluster.length >= 2 || nextCluster.some(item => item.note.hand === 'L' || item.note.midi < 58);
            if (hasBassOrChord) {
              targetNextTime = candTime;
              break;
            }
          } else {
            const hasRightOrChord = nextCluster.length >= 2 || nextCluster.some(item => item.note.hand === 'R' || item.note.midi >= 55);
            if (hasRightOrChord) {
              targetNextTime = candTime;
              break;
            }
          }
        }

        const maxGap = isBass ? MAX_CHORD_GAP_SEC : MAX_MELODY_GAP_SEC;
        if (targetNextTime !== null && (targetNextTime - n.time) <= maxGap) {
          const maxEnd = Math.min(targetNextTime - RELEASE_GAP_SEC, nextSamePitchTime[nIdx] - 0.03);
          const desiredDur = Math.round((maxEnd - n.time) * 1000) / 1000;
          if (n.duration <= (isBass ? 1.1 : 0.75) && desiredDur > n.duration + 0.05) {
            n.duration = Math.min(isBass ? 3.2 : 2.0, desiredDur);
            extendedCount++;
          }
        }
      }
    }

    return extendedCount;
  }

  /********************************************************************
   * NATIVE BINARY .MID FILE PARSER (ZERO EXTERNAL DEPENDENCIES)
   * - Parses tempo events (0x51) and time signature events (0x58)
   * - Generates synchronized measure & beat gridlines
   * - Uses assignHandsSmart for single-track or dominant-track MIDIs
   ********************************************************************/
  function parseStandardMidiFile(arrayBuffer, fileName) {
    const view = new DataView(arrayBuffer);
    let pos = 0;

    function readStr(len) {
      let s = '';
      for (let i = 0; i < len && pos < view.byteLength; i++) {
        s += String.fromCharCode(view.getUint8(pos++));
      }
      return s;
    }
    function readU16() { const v = view.getUint16(pos, false); pos += 2; return v; }
    function readU32() { const v = view.getUint32(pos, false); pos += 4; return v; }
    function readVarLen() {
      let val = 0;
      for (let i = 0; i < 4 && pos < view.byteLength; i++) {
        const b = view.getUint8(pos++);
        val = (val << 7) | (b & 0x7f);
        if ((b & 0x80) === 0) break;
      }
      return val;
    }

    if (readStr(4) !== 'MThd') throw new Error('Invalid MIDI header');
    const hdrLen = readU32();
    const format = readU16();
    const numTracks = readU16();
    const division = readU16();
    pos = 8 + hdrLen;

    const ticksPerBeat = (division & 0x8000) ? 480 : (division || 480);
    const tempoEvents = [{ tick: 0, usPerBeat: 500000 }]; // Default 120 BPM
    const timeSigEvents = [{ tick: 0, numerator: 4, denominator: 4 }]; // Default 4/4
    const rawTracks = [];
    let maxSongTick = 0;

    for (let t = 0; t < numTracks && pos < view.byteLength; t++) {
      const chunkId = readStr(4);
      const chunkLen = readU32();
      const trackEnd = Math.min(view.byteLength, pos + chunkLen);
      if (chunkId !== 'MTrk') {
        pos = trackEnd;
        continue;
      }

      let tick = 0;
      let runningStatus = 0;
      const activeNotes = {};
      const trackNotes = [];
      const pedalEvents = [];

      while (pos < trackEnd) {
        tick += readVarLen();
        if (pos >= trackEnd) break;

        let status = view.getUint8(pos);
        if (status < 0x80) {
          status = runningStatus;
        } else {
          pos++;
          if (status < 0xf0) runningStatus = status;
        }

        const eventType = status & 0xf0;
        const channel = status & 0x0f;

        if (eventType === 0x90 || eventType === 0x80) {
          const noteNum = view.getUint8(pos++);
          const vel = view.getUint8(pos++);
          const isNoteOn = (eventType === 0x90 && vel > 0);
          const key = channel + '_' + noteNum;

          if (isNoteOn) {
            if (activeNotes[key]) {
              const prevObj = activeNotes[key];
              const prevEnd = Math.max(prevObj.startTick + 15, tick - 5);
              if (prevEnd > maxSongTick) maxSongTick = prevEnd;
              trackNotes.push({
                midi: noteNum,
                startTick: prevObj.startTick,
                endTick: prevEnd,
                velocity: prevObj.vel
              });
            }
            activeNotes[key] = { startTick: tick, vel: vel / 127, midi: noteNum };
          } else if (activeNotes[key]) {
            const startObj = activeNotes[key];
            delete activeNotes[key];
            const endTick = Math.max(startObj.startTick + 20, tick);
            if (endTick > maxSongTick) maxSongTick = endTick;
            trackNotes.push({
              midi: noteNum,
              startTick: startObj.startTick,
              endTick: endTick,
              velocity: startObj.vel
            });
          }
        } else if (eventType === 0xb0) {
          const ccNum = view.getUint8(pos++);
          const ccVal = view.getUint8(pos++);
          if (ccNum === 64) {
            pedalEvents.push({ tick: tick, down: ccVal >= 64 });
          }
        } else if (eventType === 0xa0 || eventType === 0xe0) {
          pos += 2;
        } else if (eventType === 0xc0 || eventType === 0xd0) {
          pos += 1;
        } else if (status === 0xff) {
          const metaType = view.getUint8(pos++);
          const metaLen = readVarLen();
          if (metaType === 0x51 && metaLen === 3 && pos + 3 <= trackEnd) {
            const us = (view.getUint8(pos) << 16) | (view.getUint8(pos + 1) << 8) | view.getUint8(pos + 2);
            tempoEvents.push({ tick, usPerBeat: us });
          } else if (metaType === 0x58 && metaLen >= 2 && pos + 2 <= trackEnd) {
            const num = Math.max(1, view.getUint8(pos));
            const denPow = view.getUint8(pos + 1);
            const den = Math.max(1, Math.pow(2, denPow));
            timeSigEvents.push({ tick, numerator: num, denominator: den });
          }
          pos += metaLen;
        } else if (status === 0xf0 || status === 0xf7) {
          const sysLen = readVarLen();
          pos += sysLen;
        } else {
          pos++;
        }
      }

      if (pedalEvents.length > 0 && trackNotes.length > 0) {
        pedalEvents.sort((a, b) => a.tick - b.tick);
        const pedalIntervals = [];
        let pDownTick = null;
        for (let pIdx = 0; pIdx < pedalEvents.length; pIdx++) {
          const pe = pedalEvents[pIdx];
          if (pe.down && pDownTick === null) {
            pDownTick = pe.tick;
          } else if (!pe.down && pDownTick !== null) {
            if (pe.tick > pDownTick) {
              pedalIntervals.push([pDownTick, pe.tick]);
            }
            pDownTick = null;
          }
        }

        if (pedalIntervals.length > 0) {
          trackNotes.sort((a, b) => a.startTick - b.startTick || a.midi - b.midi);
          const nextSameMidiTick = new Array(trackNotes.length).fill(Infinity);
          const lastTickByMidi = {};
          for (let ti = trackNotes.length - 1; ti >= 0; ti--) {
            const m = trackNotes[ti].midi;
            if (lastTickByMidi[m] !== undefined) {
              nextSameMidiTick[ti] = lastTickByMidi[m];
            }
            lastTickByMidi[m] = trackNotes[ti].startTick;
          }

          for (let ti = 0; ti < trackNotes.length; ti++) {
            const tn = trackNotes[ti];
            for (let pi = 0; pi < pedalIntervals.length; pi++) {
              const downTick = pedalIntervals[pi][0];
              const upTick = pedalIntervals[pi][1];
              if (downTick <= tn.endTick + 20 && upTick > tn.endTick) {
                const cappedEnd = Math.min(upTick, nextSameMidiTick[ti] - 15);
                if (cappedEnd > tn.endTick) {
                  tn.endTick = cappedEnd;
                  if (tn.endTick > maxSongTick) maxSongTick = tn.endTick;
                }
                break;
              }
            }
          }
        }
      }

      if (trackNotes.length > 0) {
        rawTracks.push(trackNotes);
      }
      pos = trackEnd;
    }

    tempoEvents.sort((a, b) => a.tick - b.tick);
    timeSigEvents.sort((a, b) => a.tick - b.tick);

    function tickToSeconds(targetTick) {
      let sec = 0;
      let prevTick = 0;
      let curUs = 500000;
      for (let i = 0; i < tempoEvents.length; i++) {
        const te = tempoEvents[i];
        if (te.tick >= targetTick) break;
        sec += ((te.tick - prevTick) / ticksPerBeat) * (curUs / 1000000);
        prevTick = te.tick;
        curUs = te.usPerBeat;
      }
      sec += ((targetTick - prevTick) / ticksPerBeat) * (curUs / 1000000);
      return sec;
    }

    const finalNotes = [];
    const totalTrackNotes = rawTracks.reduce((sum, tr) => sum + tr.length, 0);
    const maxSingleTrackNotes = rawTracks.reduce((max, tr) => Math.max(max, tr.length), 0);
    const useSmartSplit = (rawTracks.length === 1) ||
      (totalTrackNotes > 0 && (maxSingleTrackNotes / totalTrackNotes) > 0.80);

    if (rawTracks.length >= 2 && !useSmartSplit) {
      const avgs = rawTracks.map(tr => tr.reduce((s, n) => s + n.midi, 0) / tr.length);
      const maxAvgIdx = avgs.indexOf(Math.max.apply(null, avgs));
      rawTracks.forEach((tr, idx) => {
        const hand = (idx === maxAvgIdx || avgs[idx] >= 60) ? 'R' : 'L';
        tr.forEach(n => {
          const t0 = tickToSeconds(n.startTick);
          const t1 = tickToSeconds(n.endTick);
          finalNotes.push({
            midi: n.midi,
            name: global.midiToName(n.midi),
            time: Math.round((t0 + 0.8) * 1000) / 1000,
            duration: Math.max(0.055, Math.round((t1 - t0) * 1000) / 1000),
            hand,
            velocity: n.velocity || 0.78
          });
        });
      });
    } else {
      rawTracks.forEach(tr => {
        tr.forEach(n => {
          const t0 = tickToSeconds(n.startTick);
          const t1 = tickToSeconds(n.endTick);
          finalNotes.push({
            midi: n.midi,
            name: global.midiToName(n.midi),
            time: Math.round((t0 + 0.8) * 1000) / 1000,
            duration: Math.max(0.055, Math.round((t1 - t0) * 1000) / 1000),
            hand: 'R',
            velocity: n.velocity || 0.78
          });
        });
      });
      assignHandsSmart(finalNotes);
    }

    if (finalNotes.length === 0) throw new Error('No playable notes in MIDI');
    finalNotes.sort((a, b) => a.time - b.time || a.midi - b.midi);
    const maxEnd = finalNotes.reduce((m, n) => Math.max(m, n.time + n.duration), 0);
    const duration = maxEnd + 2.0;

    // Generate MIDI-synchronized measure & beat gridlines
    const beats = [];
    let curTick = 0;
    let barNumber = 1;
    let tsIdx = 0;
    const endGridTick = maxSongTick + ticksPerBeat * 4;

    while (curTick <= endGridTick) {
      while (tsIdx + 1 < timeSigEvents.length && timeSigEvents[tsIdx + 1].tick <= curTick) {
        tsIdx++;
      }
      const ts = timeSigEvents[tsIdx];
      const num = Math.max(1, ts.numerator || 4);
      const den = Math.max(1, ts.denominator || 4);
      const ticksPerUnitBeat = Math.max(20, Math.round(ticksPerBeat * (4 / den)));

      for (let b = 1; b <= num; b++) {
        const beatTime = tickToSeconds(curTick) + 0.8;
        if (beatTime <= duration + 0.1) {
          beats.push({
            time: Math.round(beatTime * 1000) / 1000,
            isMeasure: b === 1,
            barNumber: barNumber,
            beatNumber: b
          });
        }
        curTick += ticksPerUnitBeat;
        if (tsIdx + 1 < timeSigEvents.length && curTick >= timeSigEvents[tsIdx + 1].tick && b < num) {
          curTick = timeSigEvents[tsIdx + 1].tick;
          break;
        }
      }
      barNumber++;
      if (beats.length > 4000) break; // Safety cap for extremely long files
    }

    const firstTempo = (tempoEvents && tempoEvents.length > 0) ? tempoEvents[0].usPerBeat : 500000;
    const parsedBpm = Math.max(30, Math.min(260, Math.round(60000000 / (firstTempo || 500000))));
    const firstTimeSig = (timeSigEvents && timeSigEvents.length > 0) ? timeSigEvents[0].numerator : 4;
    const parsedBeatsPerBar = Math.max(1, Math.min(12, Number(firstTimeSig) || 4));

    return {
      title: fileName.replace(/\.(mid|midi)$/i, ''),
      notes: finalNotes,
      duration: duration,
      beats: beats,
      bpm: parsedBpm,
      beatsPerBar: parsedBeatsPerBar,
      gridOffset: 0.8
    };
  }

  /********************************************************************
   * AUTO-SCALING OCTAVE RANGE CALCULATOR
   ********************************************************************/
  function calculateAutoRange(state) {
    const notes = state.song.notes;
    if (state.rangeMode === 'full88') {
      state.minMidi = 21;
      state.maxMidi = 108;
    } else if (state.rangeMode === 'compact') {
      let sum = 0;
      for (let i = 0; i < notes.length; i++) sum += notes[i].midi;
      const avg = notes.length ? Math.round(sum / notes.length) : 60;
      const lowC = Math.max(24, Math.floor(avg / 12) * 12 - 12);
      state.minMidi = lowC;
      state.maxMidi = Math.min(108, lowC + 24);
    } else {
      // Auto-Scale Range: exact octaves used by the song (minimum 2 octaves, max 88 keys)
      let minM = 108;
      let maxM = 21;
      for (let i = 0; i < notes.length; i++) {
        const m = notes[i].midi;
        if (m < minM) minM = m;
        if (m > maxM) maxM = m;
      }
      if (minM > maxM) { minM = 48; maxM = 72; }

      let lowC = Math.floor(minM / 12) * 12;
      let highC = Math.ceil((maxM + 1) / 12) * 12;

      while (highC - lowC < 24) {
        if (lowC >= 36) lowC -= 12;
        if (highC - lowC < 24 && highC <= 96) highC += 12;
      }
      state.minMidi = Math.max(21, lowC);
      state.maxMidi = Math.min(108, highC);
    }

    const badge = document.getElementById('activeRangeBadge');
    if (badge) {
      badge.textContent = global.midiToName(state.minMidi) + '–' + global.midiToName(state.maxMidi);
    }
  }

  global.assignHandsSmart = assignHandsSmart;
  global.extendChordLegato = extendChordLegato;
  global.parseStandardMidiFile = parseStandardMidiFile;
  global.calculateAutoRange = calculateAutoRange;
})(window);
