/* ==========================================================================
 * KeyFlow - Client-Side Chord Sheet & Synced Lyrics Importer
 * Parses chord-over-lyrics charts, barline chord progressions (| Am F |),
 * explicit chord durations (e.g., G (1.5 bars), Am (2b)), ChordPro inline
 * tags ([C]When I [G]find), and lyric-less instrumental sections ([Intro],
 * [Solo]), generating smooth voice-led piano notes + timed lyric lines.
 * ========================================================================== */

(function (global) {
  'use strict';

  const NOTE_SHARPS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const NOTE_FLATS  = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'];
  const LETTER_PC   = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };

  const CHORD_QUALITIES = {
    '': [0, 4, 7],
    'maj': [0, 4, 7],
    'M': [0, 4, 7],
    'Δ': [0, 4, 7],

    'm': [0, 3, 7],
    'min': [0, 3, 7],
    '-': [0, 3, 7],

    '5': [0, 7],

    'sus4': [0, 5, 7],
    'sus': [0, 5, 7],
    '4': [0, 5, 7],
    'sus2': [0, 2, 7],
    '2': [0, 2, 7],

    'dim': [0, 3, 6],
    '°': [0, 3, 6],
    'o': [0, 3, 6],
    'dim7': [0, 3, 6, 9],
    '°7': [0, 3, 6, 9],
    'o7': [0, 3, 6, 9],

    'm7b5': [0, 3, 6, 10],
    'min7b5': [0, 3, 6, 10],
    '-7b5': [0, 3, 6, 10],
    'ø': [0, 3, 6, 10],
    'ø7': [0, 3, 6, 10],

    'aug': [0, 4, 8],
    '+': [0, 4, 8],
    'aug7': [0, 4, 8, 10],
    '+7': [0, 4, 8, 10],
    '7#5': [0, 4, 8, 10],
    '7+5': [0, 4, 8, 10],
    '7b5': [0, 4, 6, 10],

    '6': [0, 4, 7, 9],
    'm6': [0, 3, 7, 9],
    'min6': [0, 3, 7, 9],
    '-6': [0, 3, 7, 9],
    '6/9': [0, 4, 7, 9, 14],
    '69': [0, 4, 7, 9, 14],
    '6add9': [0, 4, 7, 9, 14],

    'add9': [0, 4, 7, 14],
    'add2': [0, 4, 7, 14],
    '(add9)': [0, 4, 7, 14],
    'madd9': [0, 3, 7, 14],
    'madd2': [0, 3, 7, 14],
    'minadd9': [0, 3, 7, 14],
    'm(add9)': [0, 3, 7, 14],
    'add4': [0, 4, 5, 7],

    '7': [0, 4, 7, 10],
    'dom7': [0, 4, 7, 10],
    'maj7': [0, 4, 7, 11],
    'M7': [0, 4, 7, 11],
    'Δ7': [0, 4, 7, 11],
    'ma7': [0, 4, 7, 11],
    'm7': [0, 3, 7, 10],
    'min7': [0, 3, 7, 10],
    '-7': [0, 3, 7, 10],
    'mmaj7': [0, 3, 7, 11],
    'mMaj7': [0, 3, 7, 11],
    'm(maj7)': [0, 3, 7, 11],
    'minmaj7': [0, 3, 7, 11],
    'min(maj7)': [0, 3, 7, 11],
    '-maj7': [0, 3, 7, 11],
    '7sus4': [0, 5, 7, 10],
    '7sus': [0, 5, 7, 10],
    'sus7': [0, 5, 7, 10],

    '9': [0, 4, 10, 14],
    'maj9': [0, 4, 11, 14],
    'M9': [0, 4, 11, 14],
    'Δ9': [0, 4, 11, 14],
    'm9': [0, 3, 10, 14],
    'min9': [0, 3, 10, 14],
    '-9': [0, 3, 10, 14],
    '9sus4': [0, 5, 10, 14],
    '9sus': [0, 5, 10, 14],
    '7b9': [0, 4, 10, 13],
    '7#9': [0, 4, 10, 15],

    '11': [0, 5, 10, 14],
    'm11': [0, 3, 7, 10, 14],
    'min11': [0, 3, 7, 10, 14],
    '13': [0, 4, 9, 10],
    'maj13': [0, 4, 9, 11],
    'M13': [0, 4, 9, 11],
    'm13': [0, 3, 9, 10],
    'min13': [0, 3, 9, 10]
  };

  const SECTION_KEYWORD_RE = /\b(intro|verse|pre-?chorus|chorus|bridge|solo|outro|interlude|instrumental|refrain|tag|hook|coda|turnaround|ending|vamp|part|break)\b/i;

  const SAMPLE_CHORD_SHEET = [
    'Title: Starlight Ballad (Demo)',
    'BPM: 86',
    'Time: 4/4',
    '',
    '[Intro]',
    '| C | G | Am F | G (1.5 bars) C (0.5 bars) |',
    '',
    '[Verse 1]',
    'C                    G',
    'City lights fade in the evening rain',
    'Am                   F',
    'Quiet hands find the melody again',
    'C                    G (1.5 bars)     F (0.5 bars)',
    'Holding onto every golden sound until dawn breaks through',
    '',
    '[Chorus]',
    'F          C         G          Am',
    'Let the chords carry you across the dark',
    'F          C         G (2b)     C (2b)',
    'Every bar comes alive beneath your hands',
    '',
    '[Solo]',
    '| Am | F | C | G (1.5 bars) | C |'
  ].join('\n');

  function formatPitchName(pc, preferFlats) {
    const idx = ((pc % 12) + 12) % 12;
    return preferFlats ? NOTE_FLATS[idx] : NOTE_SHARPS[idx];
  }

  function formatMidiName(midi) {
    if (typeof global.midiToName === 'function') {
      return global.midiToName(midi);
    }
    const m = Math.max(21, Math.min(108, Math.round(midi)));
    const pc = ((m % 12) + 12) % 12;
    const oct = Math.floor(m / 12) - 1;
    return NOTE_SHARPS[pc] + oct;
  }

  function noteToPc(letter, accidental) {
    const base = LETTER_PC[letter];
    if (typeof base !== 'number') return -1;
    let delta = 0;
    if (accidental === '#') delta = 1;
    else if (accidental === 'b') delta = -1;
    return ((base + delta) % 12 + 12) % 12;
  }

  function parseChordSymbol(rawToken) {
    if (!rawToken || typeof rawToken !== 'string') return null;
    const cleaned = rawToken
      .trim()
      .replace(/♯/g, '#')
      .replace(/♭/g, 'b')
      .replace(/\*+$/g, '');

    if (!cleaned) return null;

    const upper = cleaned.toUpperCase();
    if (upper === 'N.C.' || upper === 'NC' || upper === 'N.C' || upper === 'REST' || cleaned === '-') {
      return {
        isRest: true,
        isRepeat: false,
        symbol: 'N.C.',
        rootPc: -1,
        bassPc: -1,
        quality: 'rest',
        intervals: [],
        preferFlats: false
      };
    }

    if (cleaned === '%') {
      return {
        isRest: false,
        isRepeat: true,
        symbol: '%',
        rootPc: -1,
        bassPc: -1,
        quality: 'repeat',
        intervals: [],
        preferFlats: false
      };
    }

    const m = /^([A-G])([#b]?)(.*?)(?:\/([A-G])([#b]?))?$/.exec(cleaned);
    if (!m) return null;

    const rootLetter = m[1];
    const rootAcc = m[2] || '';
    const qualityRaw = m[3] || '';
    const bassLetter = m[4] || '';
    const bassAcc = m[5] || '';

    const intervals = CHORD_QUALITIES[qualityRaw];
    if (!intervals) return null;

    const rootPc = noteToPc(rootLetter, rootAcc);
    const bassPc = bassLetter ? noteToPc(bassLetter, bassAcc) : rootPc;
    if (rootPc < 0 || bassPc < 0) return null;

    const preferFlats = rootAcc === 'b' || bassAcc === 'b' || rootPc === 5 || rootPc === 10 || rootPc === 3 || rootPc === 8;

    return {
      isRest: false,
      isRepeat: false,
      symbol: cleaned,
      rootPc: rootPc,
      bassPc: bassPc,
      quality: qualityRaw,
      intervals: intervals.slice(),
      hasSlashBass: Boolean(bassLetter),
      preferFlats: preferFlats
    };
  }

  function transposeChord(chord, semitones) {
    if (!chord || chord.isRest || chord.isRepeat) return chord;
    const shift = Math.round(Number(semitones) || 0);
    if (shift === 0) return chord;

    const newRootPc = ((chord.rootPc + shift) % 12 + 12) % 12;
    const newBassPc = ((chord.bassPc + shift) % 12 + 12) % 12;
    const preferFlats = newRootPc === 5 || newRootPc === 10 || newRootPc === 3 || newRootPc === 8;
    const rootName = formatPitchName(newRootPc, preferFlats);
    const slashPart = chord.hasSlashBass ? '/' + formatPitchName(newBassPc, preferFlats) : '';

    return {
      isRest: false,
      isRepeat: false,
      symbol: rootName + chord.quality + slashPart,
      rootPc: newRootPc,
      bassPc: newBassPc,
      quality: chord.quality,
      intervals: chord.intervals.slice(),
      hasSlashBass: chord.hasSlashBass,
      preferFlats: preferFlats
    };
  }

  function parseDurationParen(str, beatsPerBar) {
    if (!str || typeof str !== 'string') return null;
    const m = /^\(\s*(\d+(?:\.\d+)?)\s*(bars?|measures?|m|beats?|b)?\s*\)$/i.exec(str.trim());
    if (!m) return null;
    const num = parseFloat(m[1]);
    if (!isFinite(num) || num <= 0) return null;
    const unit = (m[2] || 'b').toLowerCase();
    const bpb = Math.max(1, Math.min(6, Math.round(Number(beatsPerBar) || 4)));
    let beats = (unit.startsWith('bar') || unit.startsWith('measure') || unit === 'm')
      ? num * bpb
      : num;
    beats = Math.round(beats * 4) / 4;
    return Math.max(0.25, Math.min(32, beats));
  }

  function parseMetadataLine(trimmed) {
    const titleMatch = /^(?:title|song)\s*:\s*(.+)$/i.exec(trimmed);
    if (titleMatch) return { type: 'title', value: titleMatch[1].trim() };

    const bpmMatch = /^(?:bpm|tempo)\s*:\s*(\d+(?:\.\d+)?)\b/i.exec(trimmed);
    if (bpmMatch) {
      const bpmVal = Math.round(parseFloat(bpmMatch[1]));
      if (bpmVal >= 30 && bpmVal <= 260) return { type: 'bpm', value: bpmVal };
    }

    const timeMatch = /^(?:time|meter|signature|beats)\s*:\s*(\d+)\s*(?:\/\s*\d+)?/i.exec(trimmed);
    if (timeMatch) {
      const bpb = parseInt(timeMatch[1], 10);
      if (bpb >= 1 && bpb <= 6) return { type: 'beatsPerBar', value: bpb };
    }

    const capoMatch = /^capo\s*:?\s*(\d+)\b/i.exec(trimmed);
    if (capoMatch) {
      const capoVal = parseInt(capoMatch[1], 10);
      if (capoVal >= 0 && capoVal <= 11) return { type: 'capo', value: capoVal };
    }

    return null;
  }

  function parseSectionHeader(trimmed) {
    const bracketMatch = /^\[([^\]]+)\]\s*:?\s*$/.exec(trimmed);
    if (bracketMatch) {
      const inner = bracketMatch[1].trim();
      if (!inner) return null;
      const maybeChord = parseChordSymbol(inner);
      if (!maybeChord || SECTION_KEYWORD_RE.test(inner) || /\s|\d/.test(inner)) {
        return inner;
      }
      return null;
    }

    const colonMatch = /^([A-Za-z][A-Za-z0-9\s\-']{1,28})\s*:\s*$/.exec(trimmed);
    if (colonMatch && SECTION_KEYWORD_RE.test(colonMatch[1])) {
      return colonMatch[1].trim();
    }

    return null;
  }

  function tokenizeChordLine(line, lineStartOffset, beatsPerBar) {
    const tokens = [];
    const scanner = /(\|+|\([^)]*\)|[^|\s()]+)/g;
    let m;

    while ((m = scanner.exec(line)) !== null) {
      const raw = m[1];
      const col = m.index;
      const absStart = lineStartOffset + col;
      const absEnd = absStart + raw.length;

      if (raw[0] === '|') {
        tokens.push({ kind: 'pipe', raw: raw, col: col, absStart: absStart, absEnd: absEnd });
        continue;
      }

      if (raw[0] === '(' && raw[raw.length - 1] === ')') {
        const dur = parseDurationParen(raw, beatsPerBar);
        if (dur === null) return null;
        if (tokens.length === 0 || tokens[tokens.length - 1].kind !== 'chord') {
          return null;
        }
        const prev = tokens[tokens.length - 1];
        if (prev.explicitDuration !== null) return null;
        prev.explicitDuration = dur;
        prev.durationTextStart = absStart;
        prev.durationTextEnd = absEnd;
        continue;
      }

      const parsed = parseChordSymbol(raw);
      if (!parsed) return null;
      tokens.push({
        kind: 'chord',
        raw: raw,
        parsed: parsed,
        col: col,
        absStart: absStart,
        absEnd: absEnd,
        explicitDuration: null,
        durationTextStart: null,
        durationTextEnd: null
      });
    }

    const chordTokens = tokens.filter(t => t.kind === 'chord');
    if (chordTokens.length === 0) return null;
    return tokens;
  }

  function assignLineTokenDurations(tokens, beatsPerBar, defaultChordBeats) {
    const hasPipes = tokens.some(t => t.kind === 'pipe');
    if (!hasPipes) {
      return tokens.map(t => ({
        ...t,
        durationBeats: t.explicitDuration !== null ? t.explicitDuration : defaultChordBeats
      }));
    }

    const bars = [];
    let currentBar = [];
    for (let i = 0; i < tokens.length; i++) {
      const tok = tokens[i];
      if (tok.kind === 'pipe') {
        if (currentBar.length > 0) {
          bars.push(currentBar);
          currentBar = [];
        }
      } else {
        currentBar.push(tok);
      }
    }
    if (currentBar.length > 0) bars.push(currentBar);

    const resolved = [];
    for (let b = 0; b < bars.length; b++) {
      const bar = bars[b];
      let explicitSum = 0;
      let unspecifiedCount = 0;
      for (let j = 0; j < bar.length; j++) {
        if (bar[j].explicitDuration !== null) explicitSum += bar[j].explicitDuration;
        else unspecifiedCount++;
      }

      let shareBeats = defaultChordBeats;
      if (unspecifiedCount > 0) {
        const remaining = beatsPerBar - explicitSum;
        if (remaining > 0.01) {
          shareBeats = Math.max(0.25, Math.round((remaining / unspecifiedCount) * 1000) / 1000);
        }
      }

      for (let j = 0; j < bar.length; j++) {
        const tok = bar[j];
        resolved.push({
          ...tok,
          durationBeats: tok.explicitDuration !== null ? tok.explicitDuration : shareBeats
        });
      }
    }
    return resolved;
  }

  function tryParseChordProLine(line, lineStartOffset, beatsPerBar, defaultChordBeats) {
    if (!/\[[^\]]+\]/.test(line)) return null;
    if (parseSectionHeader(line.trim())) return null;

    const tagRe = /\[([^\]]+)\]/g;
    const matches = [];
    let m;
    while ((m = tagRe.exec(line)) !== null) {
      const inner = m[1].trim();
      let chordStr = inner;
      let explicitDuration = null;

      const durParenMatch = /^(.+?)\s*(\([^)]+\))$/.exec(inner);
      if (durParenMatch) {
        const maybeDur = parseDurationParen(durParenMatch[2], beatsPerBar);
        if (maybeDur !== null) {
          chordStr = durParenMatch[1].trim();
          explicitDuration = maybeDur;
        }
      }

      const parsed = parseChordSymbol(chordStr);
      if (!parsed) return null;

      matches.push({
        parsed: parsed,
        raw: parsed.symbol,
        absStart: lineStartOffset + m.index,
        absEnd: lineStartOffset + m.index + m[0].length,
        matchIndex: m.index,
        matchEnd: m.index + m[0].length,
        explicitDuration: explicitDuration
      });
    }

    if (matches.length === 0) return null;
    const prefix = line.slice(0, matches[0].matchIndex).replace(/\s+$/g, '');
    const chords = [];
    let plainLyrics = prefix ? prefix + ' ' : '';

    for (let i = 0; i < matches.length; i++) {
      const cur = matches[i];
      const nextStart = i + 1 < matches.length ? matches[i + 1].matchIndex : line.length;
      const segText = line.slice(cur.matchEnd, nextStart);
      plainLyrics += segText;
      chords.push({
        parsed: cur.parsed,
        raw: cur.raw,
        col: cur.matchIndex,
        absStart: cur.absStart,
        absEnd: cur.absEnd,
        explicitDuration: cur.explicitDuration,
        durationTextStart: null,
        durationTextEnd: null,
        durationBeats: cur.explicitDuration !== null ? cur.explicitDuration : defaultChordBeats,
        chordProText: (i === 0 && prefix ? prefix + ' ' : '') + segText
      });
    }

    return {
      chords: chords,
      lyricText: plainLyrics.trim(),
      isChordPro: true
    };
  }

  function splitLineByColumns(lyricText, chordItems) {
    if (!lyricText || chordItems.length === 0) return [];
    const safeText = lyricText.replace(/\s+$/g, '');
    if (!safeText.trim()) return [];

    function snapColumn(col) {
      const idx = Math.max(0, Math.min(safeText.length, col));
      if (idx === 0 || idx >= safeText.length) return idx;
      if (safeText[idx - 1] === ' ' || safeText[idx] === ' ') return idx;
      for (let d = 1; d <= 4; d++) {
        if (idx - d > 0 && safeText[idx - d - 1] === ' ') return idx - d;
        if (idx + d < safeText.length && safeText[idx + d] === ' ') return idx + d + 1;
      }
      return idx;
    }

    const cuts = chordItems.map(c => snapColumn(c.col));
    for (let i = 1; i < cuts.length; i++) {
      if (cuts[i] < cuts[i - 1]) cuts[i] = cuts[i - 1];
    }

    const segments = [];
    if (cuts[0] > 0) {
      const leadText = safeText.slice(0, cuts[0]);
      if (leadText.trim()) {
        segments.push({ text: leadText, chordEventIndex: -1, startBeat: chordItems[0].startBeat, endBeat: chordItems[0].startBeat });
      }
    }

    for (let i = 0; i < chordItems.length; i++) {
      const startCol = cuts[i];
      const endCol = i + 1 < cuts.length ? cuts[i + 1] : safeText.length;
      let chunk = safeText.slice(startCol, endCol);
      if (!chunk && i === chordItems.length - 1 && segments.length === 0) {
        chunk = safeText;
      }
      segments.push({
        text: chunk,
        chordEventIndex: chordItems[i].index,
        startBeat: chordItems[i].startBeat,
        endBeat: chordItems[i].startBeat + chordItems[i].durationBeats
      });
    }
    return segments;
  }

  function scanLinesWithOffsets(text) {
    const src = String(text || '');
    const result = [];
    let pos = 0;
    const re = /\r?\n/g;
    let m;
    while ((m = re.exec(src)) !== null) {
      result.push({ line: src.slice(pos, m.index), offset: pos });
      pos = m.index + m[0].length;
    }
    result.push({ line: src.slice(pos), offset: pos });
    return result;
  }

  function parseChordSheet(text, options) {
    const opts = options || {};
    const bpm = Math.max(30, Math.min(260, Math.round(Number(opts.bpm) || 90)));
    const beatsPerBar = Math.max(1, Math.min(6, Math.round(Number(opts.beatsPerBar) || 4)));
    const defaultChordBeats = Math.max(1, Math.min(6, Math.round(Number(opts.defaultChordBeats) || beatsPerBar)));
    const beatSec = 60 / bpm;

    const rawLines = scanLinesWithOffsets(text);
    const detectedMeta = { title: '', bpm: null, beatsPerBar: null, capo: null };
    const classified = [];

    for (let i = 0; i < rawLines.length; i++) {
      const entry = rawLines[i];
      const trimmed = entry.line.trim();
      if (!trimmed) {
        classified.push({ kind: 'blank', ...entry });
        continue;
      }
      const meta = parseMetadataLine(trimmed);
      if (meta) {
        if (meta.type === 'title' && !detectedMeta.title) detectedMeta.title = meta.value;
        if (meta.type === 'bpm' && detectedMeta.bpm === null) detectedMeta.bpm = meta.value;
        if (meta.type === 'beatsPerBar' && detectedMeta.beatsPerBar === null) detectedMeta.beatsPerBar = meta.value;
        if (meta.type === 'capo' && detectedMeta.capo === null) detectedMeta.capo = meta.value;
        classified.push({ kind: 'meta', meta: meta, ...entry });
        continue;
      }
      const sectionName = parseSectionHeader(trimmed);
      if (sectionName) {
        classified.push({ kind: 'section', sectionName: sectionName, ...entry });
        continue;
      }
      const chordPro = tryParseChordProLine(entry.line, entry.offset, beatsPerBar, defaultChordBeats);
      if (chordPro) {
        classified.push({ kind: 'chordpro', chordPro: chordPro, ...entry });
        continue;
      }
      const chordTokens = tokenizeChordLine(entry.line, entry.offset, beatsPerBar);
      if (chordTokens) {
        classified.push({ kind: 'chords', chordTokens: chordTokens, ...entry });
        continue;
      }
      classified.push({ kind: 'lyric', trimmed: trimmed, ...entry });
    }

    const events = [];
    const lyricLines = [];
    const sections = [];
    let currentSection = 'Main';
    let cursorBeat = 0;
    let lastValidParsed = null;

    function resolveRepeatParsed(parsed) {
      if (parsed.isRepeat) {
        if (lastValidParsed) {
          return { ...lastValidParsed, intervals: lastValidParsed.intervals.slice(), isRepeat: false };
        }
        return { isRest: true, isRepeat: false, symbol: 'N.C.', rootPc: -1, bassPc: -1, quality: 'rest', intervals: [], preferFlats: false };
      }
      if (!parsed.isRest) lastValidParsed = parsed;
      return parsed;
    }

    for (let i = 0; i < classified.length; i++) {
      const item = classified[i];
      if (item.kind === 'section') {
        currentSection = item.sectionName;
        if (sections.length === 0 || sections[sections.length - 1] !== currentSection) {
          sections.push(currentSection);
        }
        continue;
      }

      if (item.kind === 'chordpro') {
        const lineStartBeat = cursorBeat;
        const lineEvents = [];
        const segs = [];

        for (let c = 0; c < item.chordPro.chords.length; c++) {
          const ch = item.chordPro.chords[c];
          const resolved = resolveRepeatParsed(ch.parsed);
          const evObj = {
            index: events.length,
            symbol: resolved.symbol,
            parsed: resolved,
            durationBeats: ch.durationBeats,
            startBeat: cursorBeat,
            col: ch.col,
            section: currentSection,
            lineIndex: i,
            textStart: ch.absStart,
            textEnd: ch.absEnd,
            durationTextStart: ch.durationTextStart,
            durationTextEnd: ch.durationTextEnd,
            hasExplicitDuration: ch.explicitDuration !== null,
            isChordPro: true
          };
          events.push(evObj);
          lineEvents.push(evObj);
          segs.push({
            text: ch.chordProText,
            chordEventIndex: evObj.index,
            startBeat: cursorBeat,
            endBeat: cursorBeat + ch.durationBeats
          });
          cursorBeat += ch.durationBeats;
        }

        lyricLines.push({
          section: currentSection,
          instrumental: !item.chordPro.lyricText,
          text: item.chordPro.lyricText,
          startBeat: lineStartBeat,
          endBeat: cursorBeat,
          segments: item.chordPro.lyricText ? segs : [],
          chordIndices: lineEvents.map(e => e.index)
        });
        continue;
      }

      if (item.kind === 'chords') {
        const assigned = assignLineTokenDurations(item.chordTokens, beatsPerBar, defaultChordBeats);
        const lineStartBeat = cursorBeat;
        const lineEvents = [];

        for (let c = 0; c < assigned.length; c++) {
          const tok = assigned[c];
          const resolved = resolveRepeatParsed(tok.parsed);
          const evObj = {
            index: events.length,
            symbol: resolved.symbol,
            parsed: resolved,
            durationBeats: tok.durationBeats,
            startBeat: cursorBeat,
            col: tok.col,
            section: currentSection,
            lineIndex: i,
            textStart: tok.absStart,
            textEnd: tok.absEnd,
            durationTextStart: tok.durationTextStart,
            durationTextEnd: tok.durationTextEnd,
            hasExplicitDuration: tok.explicitDuration !== null,
            isChordPro: false
          };
          events.push(evObj);
          lineEvents.push(evObj);
          cursorBeat += tok.durationBeats;
        }

        let pairedLyricText = '';
        if (i + 1 < classified.length && classified[i + 1].kind === 'lyric') {
          pairedLyricText = classified[i + 1].line;
          i++; // Consume paired lyric line
        }

        if (pairedLyricText.trim()) {
          lyricLines.push({
            section: currentSection,
            instrumental: false,
            text: pairedLyricText.trim(),
            startBeat: lineStartBeat,
            endBeat: cursorBeat,
            segments: splitLineByColumns(pairedLyricText, lineEvents),
            chordIndices: lineEvents.map(e => e.index)
          });
        } else {
          lyricLines.push({
            section: currentSection,
            instrumental: true,
            text: '',
            startBeat: lineStartBeat,
            endBeat: cursorBeat,
            segments: [],
            chordIndices: lineEvents.map(e => e.index)
          });
        }
      }
    }

    const totalBeats = Math.round(cursorBeat * 1000) / 1000;
    const totalBars = totalBeats > 0 ? Math.ceil(totalBeats / beatsPerBar) : 0;

    return {
      events: events,
      lyricLines: lyricLines,
      sections: sections,
      totalBeats: totalBeats,
      totalBars: totalBars,
      beatsPerBar: beatsPerBar,
      defaultChordBeats: defaultChordBeats,
      bpm: bpm,
      estimatedDurationSec: totalBeats * beatSec,
      detectedMeta: detectedMeta
    };
  }

  function formatDurationLabel(beats, beatsPerBar) {
    const cleanBeats = Math.max(0.25, Math.round(beats * 4) / 4);
    if (beatsPerBar > 1 && cleanBeats >= beatsPerBar && (cleanBeats * 2) % beatsPerBar === 0) {
      const bars = cleanBeats / beatsPerBar;
      return bars === 1 ? '1 bar' : bars + ' bars';
    }
    return cleanBeats + 'b';
  }

  function updateChordDurationInText(text, eventObj, newBeats, beatsPerBar) {
    if (!text || !eventObj) return text;
    const label = '(' + formatDurationLabel(newBeats, beatsPerBar) + ')';

    if (eventObj.isChordPro) {
      const rawTag = text.slice(eventObj.textStart, eventObj.textEnd);
      const inner = rawTag.replace(/^\[|\]$/g, '').replace(/\s*\([^)]*\)\s*$/, '').trim();
      const updatedTag = '[' + inner + ' ' + label + ']';
      return text.slice(0, eventObj.textStart) + updatedTag + text.slice(eventObj.textEnd);
    }

    if (typeof eventObj.durationTextStart === 'number' && typeof eventObj.durationTextEnd === 'number') {
      return text.slice(0, eventObj.durationTextStart) + label + text.slice(eventObj.durationTextEnd);
    }

    return text.slice(0, eventObj.textEnd) + ' ' + label + text.slice(eventObj.textEnd);
  }

  function chooseRightHandVoicing(chord, prevRh) {
    const intervals = chord.intervals && chord.intervals.length > 0 ? chord.intervals : [0, 4, 7];
    const pitchClasses = [];
    for (let i = 0; i < intervals.length; i++) {
      const pc = ((chord.rootPc + intervals[i]) % 12 + 12) % 12;
      if (pitchClasses.indexOf(pc) === -1) pitchClasses.push(pc);
    }

    const candidates = [];
    // Build compact inversions across comfortable middle-keyboard octaves (A3..A5)
    for (let startIdx = 0; startIdx < pitchClasses.length; startIdx++) {
      const orderedPcs = [];
      for (let k = 0; k < pitchClasses.length; k++) {
        orderedPcs.push(pitchClasses[(startIdx + k) % pitchClasses.length]);
      }

      for (let baseOct = 3; baseOct <= 5; baseOct++) {
        const notes = [];
        let curMidi = (baseOct + 1) * 12 + orderedPcs[0];
        notes.push(curMidi);
        for (let k = 1; k < orderedPcs.length; k++) {
          let nextMidi = curMidi + (((orderedPcs[k] - (curMidi % 12)) % 12 + 12) % 12);
          if (nextMidi <= curMidi) nextMidi += 12;
          notes.push(nextMidi);
          curMidi = nextMidi;
        }

        const low = notes[0];
        const high = notes[notes.length - 1];
        if (low >= 55 && high <= 84 && (high - low) <= 14) {
          candidates.push(notes);
        }
      }
    }

    if (candidates.length === 0) {
      return pitchClasses.map((pc, idx) => 60 + ((pc - 0 + 12) % 12) + (idx > 2 ? 12 : 0));
    }

    const prevAvg = Array.isArray(prevRh) && prevRh.length > 0
      ? prevRh.reduce((s, m) => s + m, 0) / prevRh.length
      : 66;

    let best = candidates[0];
    let bestScore = Infinity;

    for (let i = 0; i < candidates.length; i++) {
      const cand = candidates[i];
      const candAvg = cand.reduce((s, m) => s + m, 0) / cand.length;
      let moveCost = Math.abs(candAvg - prevAvg) * 1.4;

      if (Array.isArray(prevRh) && prevRh.length > 0) {
        let minTopDist = 0;
        const len = Math.min(cand.length, prevRh.length);
        for (let j = 0; j < len; j++) {
          minTopDist += Math.abs(cand[j] - prevRh[j]);
        }
        moveCost += (minTopDist / len) * 0.9;
      }

      const centerPull = Math.abs(candAvg - 66) * 0.45;
      const score = moveCost + centerPull;
      if (score < bestScore) {
        bestScore = score;
        best = cand;
      }
    }

    return best.sort((a, b) => a - b);
  }

  function chooseBassMidi(chord, prevBass) {
    const pc = chord.bassPc >= 0 ? chord.bassPc : chord.rootPc;
    const options = [];
    for (let m = 36 + pc; m <= 55; m += 12) {
      if (m >= 36 && m <= 55) options.push(m);
    }
    if (options.length === 0) return 48;
    const target = typeof prevBass === 'number' ? prevBass : 45;
    let best = options[0];
    let bestDist = Infinity;
    for (let i = 0; i < options.length; i++) {
      const d = Math.abs(options[i] - target) + Math.abs(options[i] - 44) * 0.25;
      if (d < bestDist) {
        bestDist = d;
        best = options[i];
      }
    }
    return best;
  }

  function assignRhFingers(rhNotes) {
    if (rhNotes.length === 1) return [3];
    if (rhNotes.length === 2) return [1, 5];
    if (rhNotes.length === 3) {
      const spanTop = rhNotes[2] - rhNotes[1];
      return spanTop >= 5 ? [1, 2, 5] : [1, 3, 5];
    }
    if (rhNotes.length === 4) return [1, 2, 3, 5];
    return rhNotes.map((_, idx) => Math.min(5, idx + 1));
  }

  function pushChordNote(notes, midi, timeSec, durationSec, hand, velocity, finger) {
    const safeMidi = Math.max(21, Math.min(108, Math.round(midi)));
    const safeTime = Math.max(0, Math.round(timeSec * 1000) / 1000);
    const safeDur = Math.max(0.08, Math.round(durationSec * 1000) / 1000);
    notes.push({
      midi: safeMidi,
      name: formatMidiName(safeMidi),
      time: safeTime,
      duration: safeDur,
      hand: hand === 'L' ? 'L' : 'R',
      velocity: Math.max(0.25, Math.min(0.95, velocity)),
      finger: finger
    });
  }

  function resolveSamePitchOverlaps(notes) {
    notes.sort((a, b) => a.time - b.time || a.midi - b.midi);
    const lastByMidi = {};
    for (let i = 0; i < notes.length; i++) {
      const n = notes[i];
      const prev = lastByMidi[n.midi];
      if (prev && prev.time + prev.duration > n.time - 0.025) {
        prev.duration = Math.max(0.06, Math.round((n.time - 0.025 - prev.time) * 1000) / 1000);
      }
      lastByMidi[n.midi] = n;
    }
    return notes;
  }

  function buildSongFromChordChart(text, options) {
    const opts = options || {};
    const titleInput = String(opts.title || '').trim();
    const bpm = Math.max(30, Math.min(260, Math.round(Number(opts.bpm) || 90)));
    const beatsPerBar = Math.max(1, Math.min(6, Math.round(Number(opts.beatsPerBar) || 4)));
    const defaultChordBeats = Math.max(1, Math.min(6, Math.round(Number(opts.defaultChordBeats) || beatsPerBar)));
    const capo = Math.max(0, Math.min(11, Math.round(Number(opts.capo) || 0)));
    const pattern = ['block', 'ballad', 'arpeggio', 'groove'].includes(opts.pattern) ? opts.pattern : 'ballad';

    const parsed = parseChordSheet(text, {
      bpm: bpm,
      beatsPerBar: beatsPerBar,
      defaultChordBeats: defaultChordBeats
    });

    if (!parsed.events || parsed.events.length === 0) {
      throw new Error('No chords were detected. Use chord symbols like C, G (1.5 bars), Am7, F/C, or | Am F |.');
    }

    const beatSec = 60 / bpm;
    const leadInSec = 1.2;
    const notes = [];
    let prevRh = null;
    let prevBass = null;

    const transposedEvents = parsed.events.map(ev => {
      const tChord = transposeChord(ev.parsed, capo);
      return {
        ...ev,
        parsed: tChord,
        symbol: tChord.symbol
      };
    });

    for (let idx = 0; idx < transposedEvents.length; idx++) {
      const ev = transposedEvents[idx];
      if (ev.parsed.isRest) continue;

      const startSec = leadInSec + ev.startBeat * beatSec;
      const chordBeats = ev.durationBeats;
      const rhNotes = chooseRightHandVoicing(ev.parsed, prevRh);
      const rhFingers = assignRhFingers(rhNotes);
      const bassMidi = chooseBassMidi(ev.parsed, prevBass);
      prevRh = rhNotes;
      prevBass = bassMidi;

      const octaveBass = bassMidi + 12 < rhNotes[0] ? bassMidi + 12 : null;
      const fifthBass = bassMidi + 7 < rhNotes[0] - 1 ? bassMidi + 7 : null;

      if (pattern === 'block') {
        const dur = Math.max(0.2, (chordBeats - 0.1) * beatSec);
        pushChordNote(notes, bassMidi, startSec, dur, 'L', 0.72, 5);
        if (octaveBass !== null) pushChordNote(notes, octaveBass, startSec, dur, 'L', 0.64, 1);
        for (let r = 0; r < rhNotes.length; r++) {
          pushChordNote(notes, rhNotes[r], startSec, dur, 'R', 0.76, rhFingers[r]);
        }
      } else if (pattern === 'ballad') {
        // Re-strike LH bass cleanly per bar boundary for long holds (e.g. 1.5 bars)
        for (let lb = 0; lb < chordBeats - 0.01; lb += beatsPerBar) {
          const spanBeats = Math.min(beatsPerBar, chordBeats - lb);
          const lhDur = Math.max(0.2, (spanBeats - 0.08) * beatSec);
          const lhTime = startSec + lb * beatSec;
          pushChordNote(notes, bassMidi, lhTime, lhDur, 'L', 0.74, 5);
          if (octaveBass !== null) {
            pushChordNote(notes, octaveBass, lhTime, lhDur, 'L', 0.64, 1);
          }
        }
        // Quarter-note RH pulses through the chord's duration
        for (let b = 0; b < chordBeats - 0.01; b += 1) {
          const stepBeats = Math.min(1, chordBeats - b);
          const rhDur = Math.max(0.16, stepBeats * 0.86 * beatSec);
          const hitTime = startSec + b * beatSec;
          const vel = (b % beatsPerBar === 0) ? 0.78 : 0.65;
          for (let r = 0; r < rhNotes.length; r++) {
            pushChordNote(notes, rhNotes[r], hitTime, rhDur, 'R', vel, rhFingers[r]);
          }
        }
      } else if (pattern === 'arpeggio') {
        // LH root + fifth alternating every bar / half-bar
        for (let lb = 0; lb < chordBeats - 0.01; lb += Math.max(1, beatsPerBar / 2)) {
          const isDown = (Math.round(lb * 2) % Math.max(2, beatsPerBar * 2)) === 0;
          const midi = (!isDown && fifthBass !== null) ? fifthBass : bassMidi;
          const stepSpan = Math.min(Math.max(1, beatsPerBar / 2), chordBeats - lb);
          pushChordNote(notes, midi, startSec + lb * beatSec, stepSpan * 0.9 * beatSec, 'L', isDown ? 0.74 : 0.64, isDown ? 5 : 2);
        }
        // RH rolling eighth-note arpeggio up and down
        const seq = rhNotes.length >= 3
          ? rhNotes.concat(rhNotes.slice(1, rhNotes.length - 1).reverse())
          : rhNotes;
        const fingerSeq = rhFingers.length >= 3
          ? rhFingers.concat(rhFingers.slice(1, rhFingers.length - 1).reverse())
          : rhFingers;
        let stepIdx = 0;
        for (let b = 0; b < chordBeats - 0.01; b += 0.5) {
          const idxInSeq = stepIdx % seq.length;
          const durBeats = Math.min(0.46, chordBeats - b);
          pushChordNote(
            notes,
            seq[idxInSeq],
            startSec + b * beatSec,
            Math.max(0.12, durBeats * beatSec),
            'R',
            idxInSeq === 0 ? 0.78 : 0.66,
            fingerSeq[idxInSeq]
          );
          stepIdx++;
        }
      } else {
        // 'groove': syncopated pop comping inside each beatsPerBar window
        const rhOffsets = beatsPerBar >= 4
          ? [{ b: 0, d: 1.25 }, { b: 1.5, d: 1.25 }, { b: 3.0, d: 0.85 }]
          : (beatsPerBar === 3
            ? [{ b: 0, d: 1.25 }, { b: 1.5, d: 0.85 }, { b: 2.5, d: 0.45 }]
            : [{ b: 0, d: 0.85 }, { b: 1.0, d: 0.85 }]);
        const lhOffsets = beatsPerBar >= 4
          ? [{ b: 0, d: 2.2 }, { b: 2.5, d: 1.25 }]
          : (beatsPerBar === 3
            ? [{ b: 0, d: 1.8 }, { b: 2.0, d: 0.9 }]
            : [{ b: 0, d: Math.max(0.5, beatsPerBar - 0.15) }]);

        for (let barBase = 0; barBase < chordBeats - 0.01; barBase += beatsPerBar) {
          for (let l = 0; l < lhOffsets.length; l++) {
            const lb = barBase + lhOffsets[l].b;
            if (lb >= chordBeats - 0.05) continue;
            const lDur = Math.min(lhOffsets[l].d, chordBeats - lb);
            pushChordNote(notes, bassMidi, startSec + lb * beatSec, lDur * beatSec, 'L', l === 0 ? 0.76 : 0.66, 5);
          }
          for (let rIdx = 0; rIdx < rhOffsets.length; rIdx++) {
            const rb = barBase + rhOffsets[rIdx].b;
            if (rb >= chordBeats - 0.05) continue;
            const rDur = Math.min(rhOffsets[rIdx].d, chordBeats - rb);
            const hitT = startSec + rb * beatSec;
            const vel = rIdx === 0 ? 0.8 : 0.7;
            for (let k = 0; k < rhNotes.length; k++) {
              pushChordNote(notes, rhNotes[k], hitT, rDur * beatSec, 'R', vel, rhFingers[k]);
            }
          }
        }
      }
    }

    resolveSamePitchOverlaps(notes);
    if (notes.length === 0) {
      throw new Error('Only N.C. (rests) were found. Add at least one chord to generate playable notes.');
    }

    const lyrics = parsed.lyricLines.map(line => ({
      section: line.section || '',
      instrumental: !!line.instrumental,
      text: line.text || '',
      time: Math.round((leadInSec + line.startBeat * beatSec) * 1000) / 1000,
      endTime: Math.round((leadInSec + line.endBeat * beatSec) * 1000) / 1000,
      segments: (line.segments || []).map(seg => {
        const ev = seg.chordEventIndex >= 0 ? transposedEvents[seg.chordEventIndex] : null;
        return {
          text: seg.text || '',
          chordSymbol: ev ? ev.symbol : '',
          time: Math.round((leadInSec + seg.startBeat * beatSec) * 1000) / 1000,
          endTime: Math.round((leadInSec + seg.endBeat * beatSec) * 1000) / 1000
        };
      }),
      chordTimeline: (line.chordIndices || []).map(idx => {
        const ev = transposedEvents[idx];
        return {
          symbol: ev ? ev.symbol : '',
          time: ev ? Math.round((leadInSec + ev.startBeat * beatSec) * 1000) / 1000 : 0,
          endTime: ev ? Math.round((leadInSec + (ev.startBeat + ev.durationBeats) * beatSec) * 1000) / 1000 : 0
        };
      })
    }));

    const duration = Math.max(4.0, Math.round((leadInSec + parsed.totalBeats * beatSec + 1.0) * 100) / 100);
    const beats = typeof global.generateBeatGrid === 'function'
      ? global.generateBeatGrid(duration, beatSec, beatsPerBar, leadInSec)
      : [];

    const finalTitle = (titleInput || parsed.detectedMeta.title || 'Chord Sheet Song').slice(0, 100);
    const songId = opts.existingId || ('chords_' + Date.now() + '_' + Math.random().toString(36).slice(2, 6));

    return {
      id: songId,
      title: finalTitle,
      notes: notes,
      duration: duration,
      beats: beats,
      bpm: bpm,
      gridOffset: leadInSec,
      beatsPerBar: beatsPerBar,
      beatType: 4,
      isCustom: true,
      source: 'chords',
      createdAt: opts.createdAt || Date.now(),
      lyrics: lyrics,
      chordSheet: {
        text: String(text || ''),
        bpm: bpm,
        beatsPerBar: beatsPerBar,
        defaultChordBeats: defaultChordBeats,
        capo: capo,
        pattern: pattern
      }
    };
  }

  global.chordImporter = {
    SAMPLE_CHORD_SHEET: SAMPLE_CHORD_SHEET,
    parseChordSymbol: parseChordSymbol,
    transposeChord: transposeChord,
    parseDurationParen: parseDurationParen,
    parseChordSheet: parseChordSheet,
    formatDurationLabel: formatDurationLabel,
    updateChordDurationInText: updateChordDurationInText,
    buildSongFromChordChart: buildSongFromChordChart
  };
})(typeof window !== 'undefined' ? window : globalThis);
