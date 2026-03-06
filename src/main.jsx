import React, { useState, useEffect, useRef } from 'react'
import ReactDOM from 'react-dom/client'
import './index.css'


    

    // Firebase config
    const firebaseConfig = {
      apiKey: "AIzaSyCbfXAy3Z_Hs_mQRNHCLk30Ext4sq3k-jA",
      authDomain: "domino-pernambuco.firebaseapp.com",
      databaseURL: "https://domino-pernambuco-default-rtdb.firebaseio.com",
      projectId: "domino-pernambuco",
      storageBucket: "domino-pernambuco.firebasestorage.app",
      messagingSenderId: "464393298578",
      appId: "1:464393298578:web:f5781c8ccc0edffed7fc4b"
    };

    firebase.initializeApp(firebaseConfig);
    const db = firebase.database();

    // === Sound Effects (Web Audio API — no external files) ===
    let _audioCtx = null;
    function playSound(type) {
      try {
        if (!_audioCtx) _audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const ctx = _audioCtx;
        const now = ctx.currentTime;

        if (type === 'join') {
          // Two-note ascending chime (~200ms)
          [523.25, 659.25].forEach((freq, i) => {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'sine';
            osc.frequency.value = freq;
            gain.gain.setValueAtTime(0.18, now + i * 0.1);
            gain.gain.exponentialRampToValueAtTime(0.001, now + i * 0.1 + 0.15);
            osc.connect(gain).connect(ctx.destination);
            osc.start(now + i * 0.1);
            osc.stop(now + i * 0.1 + 0.15);
          });
        } else if (type === 'start') {
          // Three-note ascending fanfare (~400ms)
          [523.25, 659.25, 783.99].forEach((freq, i) => {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'triangle';
            osc.frequency.value = freq;
            gain.gain.setValueAtTime(0.22, now + i * 0.12);
            gain.gain.exponentialRampToValueAtTime(0.001, now + i * 0.12 + 0.2);
            osc.connect(gain).connect(ctx.destination);
            osc.start(now + i * 0.12);
            osc.stop(now + i * 0.12 + 0.2);
          });
        } else if (type === 'turn') {
          // Short soft knock — two quick taps
          [880, 1046.5].forEach((freq, i) => {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'sine';
            osc.frequency.value = freq;
            gain.gain.setValueAtTime(0.15, now + i * 0.08);
            gain.gain.exponentialRampToValueAtTime(0.001, now + i * 0.08 + 0.1);
            osc.connect(gain).connect(ctx.destination);
            osc.start(now + i * 0.08);
            osc.stop(now + i * 0.08 + 0.1);
          });
        }
      } catch (e) { /* ignore audio errors */ }
    }

    // === Neural Network Model (globals — must be outside App for loadNeuralModel call) ===
    const NN_STATE_DIM = 185;
    const NN_NUM_ACTIONS = 57;
    let _nnModel = null;
    let USE_NN_LEAF_VALUE = false;
    let _nnLeafStats = { calls: 0, totalMs: 0 };

    const loadNeuralModel = async (url) => {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`Failed to load model: ${resp.status}`);
      const buf = await resp.arrayBuffer();
      const view = new DataView(buf);
      const headerLen = view.getUint32(0, true);
      const headerBytes = new Uint8Array(buf, 4, headerLen);
      const header = JSON.parse(new TextDecoder().decode(headerBytes));
      const dataOffset = 4 + headerLen;
      const rawBytes = new Uint8Array(buf, dataOffset, header.total_floats * 4);
      const alignedBuf = new ArrayBuffer(rawBytes.length);
      new Uint8Array(alignedBuf).set(rawBytes);
      const floats = new Float32Array(alignedBuf);
      const w = {};
      for (const layer of header.layers) {
        w[layer.name] = { shape: layer.shape, data: floats.subarray(layer.offset, layer.offset + layer.length) };
      }
      _nnModel = { weights: w, arch: header.architecture, generation: header.generation };
      USE_NN_LEAF_VALUE = true;
      console.log(`Neural model loaded: gen ${header.generation}, ${header.total_floats.toLocaleString()} params`);
      return _nnModel;
    };

    // === SPIRAL / RECTANGULAR BOARD LAYOUT ENGINE ===
    // Spiral layout: connected chain from board center, spiraling outward.
    // Uses collision detection against ALL placed tiles — no boundary shrinking.
    // Tries multiple turn directions so tiles are never lost.
    function layoutSpiral(board, W, H, HW, VW, gap, pads, avoidTL, avoidBR) {
      const n = board.length;
      if (n === 0) return { tiles: [] };
      gap = gap || 2;
      pads = typeof pads === 'number'
        ? { top: pads, right: pads, bottom: pads, left: pads }
        : Object.assign({ top: 6, right: 6, bottom: 20, left: 6 }, pads || {});
      avoidTL = avoidTL || 0;
      avoidBR = avoidBR || 0;

      // Direction constants: 0=R, 1=D, 2=L, 3=U
      const R = 0, D = 1, L = 2, U = 3;
      const nextCW = d => (d + 1) % 4;

      const dimOf = d => (d === R || d === L)
        ? { w: HW, h: VW } : { w: VW, h: HW };
      const orientOf = d => (d === R || d === L) ? 'horizontal' : 'vertical';
      const flipOf = d => (d === L || d === U);

      const inBounds = (x, y, w, h) =>
        x >= pads.left && y >= pads.top &&
        x + w <= W - pads.right && y + h <= H - pads.bottom;

      const hitsDial = (x, y, w, h) =>
        (avoidTL > 0 && x < avoidTL && y < avoidTL) ||
        (avoidBR > 0 && x + w > W - avoidBR && y + h > H - avoidBR);

      const placed = []; // collision rects: {x, y, w, h}
      const COL_MARGIN = 1; // extra px margin to prevent visual touching
      const overlaps = (x, y, w, h) => {
        for (const p of placed) {
          if (x < p.x + p.w + COL_MARGIN && x + w + COL_MARGIN > p.x &&
              y < p.y + p.h + COL_MARGIN && y + h + COL_MARGIN > p.y) return true;
        }
        return false;
      };

      const ok = (x, y, w, h) =>
        inBounds(x, y, w, h) && !hitsDial(x, y, w, h) && !overlaps(x, y, w, h);

      // Continue straight in same direction
      const straight = (prev, dir) => {
        const { w, h } = dimOf(dir);
        if (dir === R) return { x: prev.x + prev.w + gap, y: prev.y, w, h };
        if (dir === D) return { x: prev.x, y: prev.y + prev.h + gap, w, h };
        if (dir === L) return { x: prev.x - w - gap, y: prev.y, w, h };
        return { x: prev.x, y: prev.y - h - gap, w, h }; // U
      };

      // Turn position: connect new direction to prev tile's exit edge
      const turn = (prev, oldDir, newDir) => {
        const { w, h } = dimOf(newDir);
        // Standard clockwise turns
        if (oldDir === R && newDir === D) return { x: prev.x + prev.w - w, y: prev.y + prev.h + gap, w, h };
        if (oldDir === D && newDir === L) return { x: prev.x - w - gap, y: prev.y + prev.h - h, w, h };
        if (oldDir === L && newDir === U) return { x: prev.x, y: prev.y - h - gap, w, h };
        if (oldDir === U && newDir === R) return { x: prev.x + prev.w + gap, y: prev.y, w, h };
        // Non-standard turns (fallback: try connecting at prev's exit)
        if (newDir === R) return { x: prev.x + prev.w + gap, y: prev.y + prev.h - h, w, h };
        if (newDir === D) return { x: prev.x + prev.w - w, y: prev.y + prev.h + gap, w, h };
        if (newDir === L) return { x: prev.x - w - gap, y: prev.y, w, h };
        return { x: prev.x, y: prev.y - h - gap, w, h }; // U
      };

      // --- Place first tile at top-left edge (below TL dial) ---
      const d0 = dimOf(R);
      let sx = pads.left;
      let sy = Math.max(pads.top, avoidTL);
      // If starting position still hits TL dial, push right
      if (avoidTL > 0 && sx < avoidTL && sy < avoidTL) sx = avoidTL;
      placed.push({ x: sx, y: sy, w: d0.w, h: d0.h });
      const result = [{ i: 0, x: sx, y: sy, orient: 'horizontal', flip: false }];
      let dir = R;

      // --- Place remaining tiles ---
      for (let i = 1; i < n; i++) {
        const prev = placed[i - 1];
        let done = false;

        // 1. Try continuing straight
        const s = straight(prev, dir);
        if (ok(s.x, s.y, s.w, s.h)) {
          placed.push({ x: s.x, y: s.y, w: s.w, h: s.h });
          result.push({ i, x: s.x, y: s.y, orient: orientOf(dir), flip: flipOf(dir) });
          done = true;
          continue;
        }

        // 2. Try clockwise turns (up to 3)
        let tryDir = dir;
        for (let t = 0; t < 3; t++) {
          const nd = nextCW(tryDir);
          const tp = turn(prev, dir, nd);
          if (ok(tp.x, tp.y, tp.w, tp.h)) {
            placed.push({ x: tp.x, y: tp.y, w: tp.w, h: tp.h });
            result.push({ i, x: tp.x, y: tp.y, orient: orientOf(nd), flip: flipOf(nd) });
            dir = nd;
            done = true;
            break;
          }
          tryDir = nd;
        }

        if (!done) break; // board truly full
      }

      return { tiles: result };
    }

    // === Stable component definitions (outside App to prevent remount on every render) ===
    const DominoDots = ({ value, size = 'normal', dotPxProp, containerPxProp }) => {
      const dotPx = dotPxProp || (size === 'small' ? 4 : size === 'board' ? 5 : 7);
      const containerPx = containerPxProp || (size === 'small' ? 18 : size === 'board' ? 22 : 32);
      const patterns = {
        0: [],
        1: [[1,1]],
        2: [[0,2],[2,0]],
        3: [[0,2],[1,1],[2,0]],
        4: [[0,0],[0,2],[2,0],[2,2]],
        5: [[0,0],[0,2],[1,1],[2,0],[2,2]],
        6: [[0,0],[0,2],[1,0],[1,2],[2,0],[2,2]]
      };
      const dots = patterns[value] || [];
      const pad = size === 'small' ? '12%' : '10%';
      return (
        <div style={{ width: containerPx, height: containerPx, position: 'relative' }}>
          {dots.map((pos, idx) => (
            <div
              key={idx}
              className="domino-dot"
              style={{
                width: dotPx, height: dotPx,
                position: 'absolute',
                top: pos[0] === 0 ? pad : pos[0] === 1 ? '50%' : `calc(100% - ${pad})`,
                left: pos[1] === 0 ? pad : pos[1] === 1 ? '50%' : `calc(100% - ${pad})`,
                transform: 'translate(-50%, -50%)'
              }}
            />
          ))}
        </div>
      );
    };

    const DominoTile = ({ tile, playable, onClick, hDims }) => {
      const fired = React.useRef(false);
      const handleTouch = (e) => {
        if (!playable) return;
        e.preventDefault();
        if (fired.current) return;
        fired.current = true;
        onClick();
        setTimeout(() => { fired.current = false; }, 300);
      };
      return (
        <div
          onClick={(e) => { if (!fired.current) onClick(); }}
          onTouchEnd={handleTouch}
          className={'domino-tile inline-flex flex-col ' + (playable ? 'playable' : 'unplayable')}
          style={{ width: hDims.w, height: hDims.h }}
        >
          <div className="flex-1 flex items-center justify-center" style={{ borderBottom: '1.5px solid #c8b898' }}>
            <DominoDots value={tile.left} dotPxProp={hDims.dot} containerPxProp={hDims.cont} />
          </div>
          <div className="flex-1 flex items-center justify-center">
            <DominoDots value={tile.right} dotPxProp={hDims.dot} containerPxProp={hDims.cont} />
          </div>
        </div>
      );
    };

    function App() {
      const [screen, setScreen] = useState('menu');
      const [roomCode, setRoomCode] = useState('');
      const [playerName, setPlayerName] = useState('');
      const [playerId, setPlayerId] = useState(null);
      const [playerSlot, setPlayerSlot] = useState(null); // 0 = human1, 2 = human2
      const [gameState, setGameState] = useState(null);
      const [error, setError] = useState('');
      const [inputCode, setInputCode] = useState('');
      const [choosingTile, setChoosingTile] = useState(null);
      const [showStarterChoice, setShowStarterChoice] = useState(false);
      const [roundAnnouncement, setRoundAnnouncement] = useState(null);
      const [handVisible, setHandVisible] = useState(true);
      const [humanCount, setHumanCount] = useState(1);
      const [aiDifficulty, setAiDifficulty] = useState(localStorage.getItem('domino_ai_difficulty') || 'hard');
      const setAiDiff = (v) => { setAiDifficulty(v); localStorage.setItem('domino_ai_difficulty', v); };
      const [botSpeed, setBotSpeed] = useState(localStorage.getItem('domino_bot_speed') || 'medium');
      const setBotSpd = (v) => { setBotSpeed(v); localStorage.setItem('domino_bot_speed', v); };
      const [starterCountdown, setStarterCountdown] = useState(null);
      const [moveTimer, setMoveTimer] = useState(null);
      const [roundCountdown, setRoundCountdown] = useState(null);
      const [showNextBtn, setShowNextBtn] = useState(false);
      const [flyingTile, setFlyingTile] = useState(null);
      const [animations, setAnimations] = useState(localStorage.getItem('domino_animations') !== 'off');
      const setAnim = (v) => { setAnimations(v); localStorage.setItem('domino_animations', v ? 'on' : 'off'); };
      const [animScore0, setAnimScore0] = useState(0);
      const [animScore1, setAnimScore1] = useState(0);
      const prevScore0Ref = useRef(0);
      const prevScore1Ref = useRef(0);
      const [dialPulse, setDialPulse] = useState(null); // 'team0' | 'team1' | null
      const [passedSlot, setPassedSlot] = useState(null); // slot number of player who just passed
      const [showStats, setShowStats] = useState(false);
      const playerStatsRef = useRef({});
      const lastTrackedRoundRef = useRef(null);
      const lastTrackedMatchRef = useRef(null);
      // === Player Profile System ===
      const HUMAN_PROFILES = [
        { key: 'ze', name: 'Zé', avatarType: 'image', avatarSrc: 'avatars/ze.png', color: '#f59e0b' },
        { key: 'alexandre', name: 'Alexandre', avatarType: 'image', avatarSrc: 'avatars/alexandre.png', color: '#3b82f6' },
        { key: 'carlinhos', name: 'Carlinhos', avatarType: 'image', avatarSrc: 'avatars/carlinhos.png', color: '#22c55e' },
        { key: 'bernd', name: 'Bernd', avatarType: 'image', avatarSrc: 'avatars/bernd.png', color: '#a855f7' },
      ];
      const BOT_PROFILES = [
        { key: 'dona-maria', name: 'Dona Maria', avatarType: 'initials', initials: 'DM', color: '#ef4444', bgGradient: 'linear-gradient(135deg, #ef4444, #b91c1c)' },
        { key: 'seu-joao', name: 'Seu João', avatarType: 'initials', initials: 'SJ', color: '#3b82f6', bgGradient: 'linear-gradient(135deg, #3b82f6, #1d4ed8)' },
        { key: 'toninho', name: 'Toninho', avatarType: 'initials', initials: 'To', color: '#f59e0b', bgGradient: 'linear-gradient(135deg, #f59e0b, #d97706)' },
        { key: 'cida', name: 'Cida', avatarType: 'initials', initials: 'Ci', color: '#8b5cf6', bgGradient: 'linear-gradient(135deg, #8b5cf6, #6d28d9)' },
        { key: 'bira', name: 'Bira', avatarType: 'initials', initials: 'Bi', color: '#10b981', bgGradient: 'linear-gradient(135deg, #10b981, #059669)' },
        { key: 'nene', name: 'Nenê', avatarType: 'initials', initials: 'Nê', color: '#f97316', bgGradient: 'linear-gradient(135deg, #f97316, #ea580c)' },
      ];

      const assignBotProfiles = (count) => {
        const shuffled = [...BOT_PROFILES].sort(() => Math.random() - 0.5);
        return shuffled.slice(0, count);
      };

      const profileFromPlayer = (player) => {
        if (!player) return null;
        if (!player.avatar) {
          return { name: player.name, avatarType: 'initials', initials: player.name.substring(0, 2).toUpperCase(), bgGradient: 'linear-gradient(135deg, #6b7280, #4b5563)', color: '#6b7280' };
        }
        return { name: player.name, avatarType: player.avatar.type, avatarSrc: player.avatar.src, initials: player.avatar.initials, bgGradient: player.avatar.bgGradient, color: player.avatar.color };
      };

      const Avatar = ({ profile, size = 32, noBorder = false }) => {
        if (!profile) return <div style={{ width: size, height: size, borderRadius: '50%', background: '#374151', flexShrink: 0 }} />;
        const fs = Math.max(10, Math.round(size * 0.4));
        const bdr = noBorder ? 'none' : '2px solid rgba(255,255,255,0.3)';
        if (profile.avatarType === 'image') {
          return (
            <div style={{ width: size, height: size, borderRadius: '50%', overflow: 'hidden', border: bdr, boxShadow: '0 2px 6px rgba(0,0,0,0.3)', flexShrink: 0, background: profile.color || '#666' }}>
              <img src={profile.avatarSrc} alt={profile.name} style={{ width: '100%', height: '100%', objectFit: 'cover' }} onError={(e) => { e.target.style.display = 'none'; }} />
            </div>
          );
        }
        return (
          <div style={{ width: size, height: size, borderRadius: '50%', background: profile.bgGradient || '#666', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: fs, fontWeight: 800, color: 'white', border: bdr, boxShadow: '0 2px 6px rgba(0,0,0,0.3)', flexShrink: 0, textShadow: '0 1px 2px rgba(0,0,0,0.3)' }}>
            {profile.initials || '?'}
          </div>
        );
      };

      const _berndProfile = HUMAN_PROFILES.find(p => p.key === 'bernd') || HUMAN_PROFILES[0];
      const [selectedProfile, setSelectedProfile] = useState(_berndProfile);
      const [boardSize, setBoardSizeRaw] = useState(localStorage.getItem('domino_board_size') || 'M');
      const setBoardSize = (v) => { setBoardSizeRaw(v); localStorage.setItem('domino_board_size', v); };
      const _storedLayout = localStorage.getItem('domino_tile_layout');
      const [tileLayout, setTileLayoutRaw] = useState((_storedLayout === 'spiral' || _storedLayout === 'snake') ? _storedLayout : 'spiral');
      const setTileLayout = (v) => { setTileLayoutRaw(v); localStorage.setItem('domino_tile_layout', v); };
      const [showEndBadges, setShowEndBadgesRaw] = useState(localStorage.getItem('domino_end_badges') === 'true');
      const setShowEndBadges = (v) => { setShowEndBadgesRaw(v); localStorage.setItem('domino_end_badges', v ? 'true' : 'false'); };
      const BOARD_DIMS = { S: { hw: 36, vw: 18 }, M: { hw: 42, vw: 21 }, L: { hw: 48, vw: 24 } };
      const bDims = BOARD_DIMS[boardSize] || BOARD_DIMS.M;

      const roomRef = useRef(null);
      const boardRef = useRef(null);
      const tileElRef = useRef(new Map());
      const playingRef = useRef(false);  // guard against double-click
      const [boardBox, setBoardBox] = useState({ w: 0, h: 0 });

      const HUMAN_FILL_ORDER = [0, 2, 1, 3];
      const getHumanSlots = (count) => HUMAN_FILL_ORDER.slice(0, count);

      const generateRoomCode = () => {
        const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
        let code = '';
        for (let i = 0; i < 5; i++) {
          code += chars[Math.floor(Math.random() * chars.length)];
        }
        return code;
      };

      const generatePlayerId = () => {
        return 'player_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
      };

      const createDeck = () => {
        const deck = [];
        for (let i = 0; i <= 6; i++) {
          for (let j = i; j <= 6; j++) {
            deck.push({ left: i, right: j, id: i + '-' + j });
          }
        }
        return deck;
      };

      const shuffleDeck = (deck) => {
        const shuffled = [...deck];
        for (let i = shuffled.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
        return shuffled;
      };

      const createRoom = async () => {
        if (!selectedProfile) {
          setError('Selecione seu perfil!');
          return;
        }

        const code = generateRoomCode();
        const pid = generatePlayerId();

        const humanSlots = getHumanSlots(humanCount);
        const botProfiles = assignBotProfiles(4 - humanCount);
        let botIdx = 0;
        const players = {};
        for (let s = 0; s < 4; s++) {
          if (s === 0) {
            players[s] = { name: selectedProfile.name, id: pid, connected: true, isHuman: true, avatar: { type: selectedProfile.avatarType, src: selectedProfile.avatarSrc, color: selectedProfile.color, key: selectedProfile.key } };
          } else if (humanSlots.includes(s)) {
            players[s] = null; // waiting for human to join
          } else {
            const bp = botProfiles[botIdx++];
            players[s] = { name: bp.name, id: 'bot-' + bp.key, connected: true, isHuman: false, avatar: { type: 'initials', initials: bp.initials, bgGradient: bp.bgGradient, color: bp.color, key: bp.key } };
          }
        }

        const initialState = {
          roomCode: code,
          config: { humanCount: humanCount, humanSlots: humanSlots, aiDifficulty: aiDifficulty, botSpeed: botSpeed },
          players: players,
          gameStarted: false,
          gameEnded: false,
          waitingForStarterChoice: false,
          lastWinningTeam: null,
          createdAt: Date.now()
        };

        try {
          await db.ref('rooms/' + code).set(initialState);
          setRoomCode(code);
          setPlayerId(pid);
          setPlayerSlot(0);
          setScreen('lobby');
          
          roomRef.current = db.ref('rooms/' + code);
          let _prevPlayerCount = Object.values(initialState.players).filter(p => p).length;
          let _prevGameStarted = false;
          roomRef.current.on('value', (snapshot) => {
            const data = snapshot.val();
            if (data) {
              setGameState(data);
              // Detect new player joining
              const curCount = data.players ? Object.values(data.players).filter(p => p).length : 0;
              if (curCount > _prevPlayerCount) playSound('join');
              _prevPlayerCount = curCount;
              // Detect game start
              if (data.gameStarted && !data.gameEnded) {
                if (!_prevGameStarted) playSound('start');
                setScreen('game');
              }
              _prevGameStarted = !!(data.gameStarted && !data.gameEnded);
              if (data.waitingForStarterChoice) {
                setShowStarterChoice(true);
              } else {
                setShowStarterChoice(false);
              }
            }
          });

          roomRef.current.child('players/0/connected').onDisconnect().set(false);
        } catch (err) {
          setError('Erro ao criar sala: ' + err.message);
        }
      };

      const joinRoom = async () => {
        if (!selectedProfile) {
          setError('Selecione seu perfil!');
          return;
        }
        if (!inputCode.trim()) {
          setError('Digite o codigo da sala!');
          return;
        }

        const code = inputCode.toUpperCase();
        const pid = generatePlayerId();

        try {
          const snapshot = await db.ref('rooms/' + code).once('value');
          const data = snapshot.val();
          
          if (!data) {
            setError('Sala nao encontrada!');
            return;
          }

          if (data.gameStarted) {
            setError('Jogo ja comecou!');
            return;
          }

          // Find next available human slot
          const cfg = data.config || { humanCount: 2, humanSlots: [0, 2] };
          const availableSlot = cfg.humanSlots.find(s => s !== 0 && (!data.players || !data.players[s]));
          if (availableSlot === undefined) {
            setError('Sala cheia!');
            return;
          }

          await db.ref('rooms/' + code + '/players/' + availableSlot).set({
            name: selectedProfile.name,
            id: pid,
            connected: true,
            isHuman: true,
            avatar: { type: selectedProfile.avatarType, src: selectedProfile.avatarSrc, color: selectedProfile.color, key: selectedProfile.key }
          });

          setRoomCode(code);
          setPlayerId(pid);
          setPlayerSlot(availableSlot);
          setScreen('lobby');

          roomRef.current = db.ref('rooms/' + code);
          let _prevPlayerCountJ = data.players ? Object.values(data.players).filter(p => p).length : 0;
          let _prevGameStartedJ = false;
          roomRef.current.on('value', (snapshot) => {
            const data = snapshot.val();
            if (data) {
              setGameState(data);
              // Detect new player joining
              const curCount = data.players ? Object.values(data.players).filter(p => p).length : 0;
              if (curCount > _prevPlayerCountJ) playSound('join');
              _prevPlayerCountJ = curCount;
              // Detect game start
              if (data.gameStarted && !data.gameEnded) {
                if (!_prevGameStartedJ) playSound('start');
                setScreen('game');
              }
              _prevGameStartedJ = !!(data.gameStarted && !data.gameEnded);
              if (data.waitingForStarterChoice) {
                setShowStarterChoice(true);
              } else {
                setShowStarterChoice(false);
              }
            }
          });

          roomRef.current.child('players/' + availableSlot + '/connected').onDisconnect().set(false);
        } catch (err) {
          setError('Erro ao entrar: ' + err.message);
        }
      };

      const startGameWithStarter = async (starterSlot) => {
        // Tiles already dealt by newRound, just set the starter
        await db.ref('rooms/' + roomCode).update({
          waitingForStarterChoice: false,
          starterChoiceDeadline: null,
          starterVotes: null,
          currentPlayer: starterSlot,
          message: gameState.players[starterSlot].name + ' comeca!'
        });
      };

      // Submit starter vote (each human on winning team votes independently)
      const submitStarterVote = async (chosenSlot) => {
        await db.ref('rooms/' + roomCode + '/starterVotes/' + playerSlot).set(chosenSlot);
      };

      // Pip count for a hand (tiebreaker: fewer pips = starts)
      const handPipCount = (hand) => hand.reduce((s, t) => s + t.left + t.right, 0);

      const startGame = async () => {
        const cfg = gameState?.config || { humanCount: 2, humanSlots: [0, 2] };
        const allReady = cfg.humanSlots.every(s => gameState?.players?.[s]);
        if (!allReady) {
          setError('Aguardando jogadores!');
          return;
        }
        setError('');

        // Deal with safety redeal if no doubles (virtually impossible: 7 doubles, 4 dormidas)
        let hands, startPlayer, highestDouble, highestDoubleTile, dormidas;
        do {
          const deck = shuffleDeck(createDeck());
          hands = [[], [], [], []];
          for (let i = 0; i < 24; i++) {
            hands[i % 4].push(deck[i]);
          }
          dormidas = deck.slice(24, 28);
          startPlayer = 0;
          highestDouble = -1;
          highestDoubleTile = null;
          for (let p = 0; p < 4; p++) {
            for (let tile of hands[p]) {
              if (tile.left === tile.right && tile.left > highestDouble) {
                highestDouble = tile.left;
                startPlayer = p;
                highestDoubleTile = tile;
              }
            }
          }
        } while (highestDoubleTile === null);

        // Remove the highest double from the starter's hand and place it on the board
        hands[startPlayer] = hands[startPlayer].filter(t => t.id !== highestDoubleTile.id);
        const nextPlayer = (startPlayer + 1) % 4;

        const game = {
          ...gameState,
          gameStarted: true,
          gameEnded: false,
          hands: hands,
          dormidas: dormidas,
          board: [highestDoubleTile],
          leftEnd: highestDoubleTile.left,
          rightEnd: highestDoubleTile.right,
          currentPlayer: nextPlayer,
          passCount: 0,
          teamScores: [0, 0],
          scoreMultiplier: 1,
          isDobrada: false,
          lastWinningTeam: null,
          waitingForStarterChoice: false,
          moveHistory: [{p: startPlayer, t: 'play', tile: highestDoubleTile}],
          message: gameState.players[startPlayer].name + ' jogou a carroca ' + highestDouble + '-' + highestDouble + '! Vez de ' + gameState.players[nextPlayer].name + '!',
          matchTarget: 6
        };

        await db.ref('rooms/' + roomCode).set(game);
      };

      const canPlayTile = (tile) => {
        if (!gameState || !gameState.board || gameState.board.length === 0) return true;
        const { leftEnd, rightEnd } = gameState;
        return tile.left === leftEnd || tile.right === leftEnd || 
               tile.left === rightEnd || tile.right === rightEnd;
      };

      const canPlayOnBothEnds = (tile) => {
        if (!gameState || !gameState.board || gameState.board.length === 0) return false;
        const { leftEnd, rightEnd } = gameState;
        if (leftEnd === rightEnd) {
          return tile.left === leftEnd && tile.right === leftEnd;
        }
        const canLeft = tile.left === leftEnd || tile.right === leftEnd;
        const canRight = tile.left === rightEnd || tile.right === rightEnd;
        return canLeft && canRight;
      };

      const couldPlayOnBothEnds = (tile, left, right) => {
        if (left === null || right === null) return false;
        if (left === right) {
          return tile.left === left && tile.right === left;
        }
        const canLeft = tile.left === left || tile.right === left;
        const canRight = tile.left === right || tile.right === right;
        return canLeft && canRight;
      };

      // === 28 immutable tiles ===
      const ALL_TILES = createDeck();

      // === Knowledge class — full belief model ported from simulator ===
      class Knowledge {
        constructor() {
          this.cantHave = [new Set(), new Set(), new Set(), new Set()];
          this.played = new Set();
          this.playsBy = [[], [], [], []];
          this.passedOn = [[], [], [], []];
          this._strengthCache = [null, null, null, null];
          this._remainingCount = [7, 7, 7, 7, 7, 7, 7];
          this.openingSuits = [null, null, null, null];
          this.sacrificeFlags = [new Set(), new Set(), new Set(), new Set()];
          this._moveCount = 0;
        }
        clone() {
          const k = new Knowledge();
          k.cantHave = this.cantHave.map(s => new Set(s));
          k.played = new Set(this.played);
          k.playsBy = this.playsBy.map(a => [...a]);
          k.passedOn = this.passedOn.map(a => [...a]);
          k._strengthCache = [null, null, null, null];
          k._remainingCount = [...this._remainingCount];
          k.openingSuits = [...this.openingSuits];
          k.sacrificeFlags = this.sacrificeFlags.map(s => new Set(s));
          k._moveCount = this._moveCount;
          return k;
        }
        recordPlay(p, t) {
          if (!this.played.has(t.id)) {
            this.played.add(t.id);
            this._remainingCount[t.left]--;
            if (t.left !== t.right) this._remainingCount[t.right]--;
          }
          if (p >= 0 && p <= 3) {
            this.playsBy[p].push(t);
            this._strengthCache[p] = null;
            if (this.openingSuits[p] === null && this.playsBy[p].length <= 2 && this._moveCount < 8) {
              this.openingSuits[p] = t.left === t.right ? t.left : t.left;
            }
            if (this.playsBy[p].length >= 2 && t.left + t.right >= 9 && t.left !== t.right) {
              this.sacrificeFlags[p].add(t.left);
              this.sacrificeFlags[p].add(t.right);
            }
          }
          this._moveCount++;
        }
        recordPass(p, lE, rE) {
          this.cantHave[p].add(lE);
          this.cantHave[p].add(rE);
          this.passedOn[p].push({ lE, rE, move: this._moveCount });
        }
        avoidanceStrength(p, n) {
          let strength = 0;
          for (const pass of this.passedOn[p]) {
            if (pass.lE === n || pass.rE === n) {
              const age = this._moveCount - (pass.move || 0);
              strength += age < 5 ? 1.0 : (age < 10 ? 0.7 : 0.4);
            }
          }
          return strength;
        }
        inferStrength(p) {
          if (this._strengthCache[p]) return this._strengthCache[p];
          const s = [0,0,0,0,0,0,0];
          for (const t of this.playsBy[p]) { s[t.left]++; if (t.left !== t.right) s[t.right]++; }
          this._strengthCache[p] = s;
          return s;
        }
        remainingWithNumber(n) { return this._remainingCount[n]; }
        deadNumbers(p) { return [...this.cantHave[p]]; }
        possibleTilesFor(p) {
          const possible = [];
          for (const t of ALL_TILES) {
            if (this.played.has(t.id)) continue;
            if (this.cantHave[p].has(t.left) || this.cantHave[p].has(t.right)) continue;
            possible.push(t);
          }
          return possible;
        }
        chicoteFor(n, myHand) {
          if (this._remainingCount[n] !== 1) return null;
          let ct = null;
          for (const t of ALL_TILES) {
            if (this.played.has(t.id)) continue;
            if (t.left === n || t.right === n) { ct = t; break; }
          }
          if (!ct) return null;
          if (myHand && myHand.some(h => h.id === ct.id)) return { tile: ct, holder: 'self', confidence: 1 };
          const elig = [];
          for (let p = 0; p < 4; p++) {
            const blocked = ct.left === ct.right
              ? this.cantHave[p].has(ct.left)
              : (this.cantHave[p].has(ct.left) && this.cantHave[p].has(ct.right));
            if (!blocked) elig.push(p);
          }
          if (elig.length === 0) return { tile: ct, holder: 'dorme', confidence: 1 };
          if (elig.length === 1) return { tile: ct, holder: elig[0], confidence: 1 };
          return { tile: ct, holder: 'unknown', candidates: elig, confidence: 1 / elig.length };
        }
        isProbablyDead(n, myIdx) {
          if (this._remainingCount[n] === 0) return true;
          for (let p = 0; p < 4; p++) {
            if (p === myIdx) continue;
            if (!this.cantHave[p].has(n)) return false;
          }
          return true;
        }
      }

      // Build Knowledge from moveHistory (for mobile game's Firebase-based state)
      const buildKnowledge = (moveHistory) => {
        const k = new Knowledge();
        for (const move of (moveHistory || [])) {
          if (move.t === 'play') {
            k.recordPlay(move.p, move.tile);
          } else if (move.t === 'pass') {
            k.recordPass(move.p, move.lE, move.rE);
          }
        }
        return k;
      };

      // === AI WEIGHTS — tunable scoring parameters ===
      const AI_WEIGHTS = {
        deadEndPenalty: 34.7, lockFavorable: 42.5, lockUnfavorable: 62.8,
        chicoteSelf: 26.9, chicotePartner: 23.4, chicoteOpponent: 23.7, chicoteDorme: 16.9,
        lockApproachGood: 13.5, lockApproachBad: 17.5, monopolyBonus: 17.5,
        boardCountGradient: 6.3, captiveEndBonus: 16, probDeadPenalty: 21.9,
      };

      // Standalone scoreWin for smartAI closing bonus
      const scoreWin = (tile, prevLE, prevRE, boardLen) => {
        const isD = tile.left === tile.right;
        const wasBoth = boardLen > 1 && couldPlayOnBothEnds(tile, prevLE, prevRE);
        if (isD && wasBoth) return { pts: 4, type: 'CRUZADA' };
        if (isD) return { pts: 2, type: 'CARROCA' };
        if (wasBoth) return { pts: 3, type: 'LA E LO' };
        return { pts: 1, type: 'NORMAL' };
      };

      // === Full smartAI — ported from simulator with all strategic features ===
      const smartAI = (hand, lE, rE, bLen, player, knowledge, matchScores, returnAll = false) => {
        const canPlay = (t) => bLen === 0 || t.left === lE || t.right === lE || t.left === rE || t.right === rE;
        const playable = hand.filter(canPlay);
        if (playable.length === 0) return returnAll ? [] : null;

        if (bLen === 0) {
          const sc = [0,0,0,0,0,0,0];
          for (const t of hand) { sc[t.left]++; if (t.left !== t.right) sc[t.right]++; }
          const scored = playable.map(t => {
            let s = sc[t.left]*10 + sc[t.right]*10 + (t.left===t.right?15:0) + (t.left+t.right)*2;
            return { tile: t, score: s, side: null };
          });
          scored.sort((a, b) => b.score - a.score);
          return returnAll ? scored : scored[0];
        }

        const partner = (player + 2) % 4;
        const opp1 = (player + 1) % 4, opp2 = (player + 3) % 4;
        const deadMul = bLen <= 4 ? 0.4 : (bLen <= 14 ? 1.0 : 1.4);

        const suitCount = new Array(7).fill(0);
        const tilesByNum = Array.from({length: 7}, () => []);
        for (const t of hand) {
          suitCount[t.left]++;
          if (t.left !== t.right) suitCount[t.right]++;
          tilesByNum[t.left].push(t);
          if (t.left !== t.right) tilesByNum[t.right].push(t);
        }

        const scored = playable.flatMap(tile => {
          const options = [];
          for (const side of ['left', 'right']) {
            const cS = side === 'left' ? (tile.left === lE || tile.right === lE) : (tile.left === rE || tile.right === rE);
            if (!cS) continue;

            let newEnd;
            if (side === 'left') newEnd = tile.left === lE ? tile.right : tile.left;
            else newEnd = tile.right === rE ? tile.left : tile.right;

            const otherEnd = side === 'left' ? rE : lE;
            let ss = 0;

            // Suit control
            const endSet = new Set([newEnd, otherEnd]);
            let myCount = 0;
            for (const n of endSet) {
              for (const x of tilesByNum[n]) { if (x.id !== tile.id) myCount++; }
            }
            if (endSet.size === 2) {
              for (const x of tilesByNum[newEnd]) {
                if (x.id !== tile.id && (x.left === otherEnd || x.right === otherEnd)) myCount--;
              }
            }
            ss += myCount * 15;

            // Blocking — opp1 plays next (higher value), opp2 after partner (lower)
            if (knowledge.cantHave[opp1].has(newEnd) && knowledge.cantHave[opp1].has(otherEnd)) ss += 35;
            if (knowledge.cantHave[opp2].has(newEnd) && knowledge.cantHave[opp2].has(otherEnd)) ss += 25;

            // Signaling inference: opening suit and sacrifice awareness
            if (knowledge.openingSuits && knowledge.openingSuits[partner] !== null) {
              const partnerSuit = knowledge.openingSuits[partner];
              if (newEnd === partnerSuit && !knowledge.cantHave[partner].has(partnerSuit)) {
                ss += 6;
              }
            }
            if (knowledge.sacrificeFlags) {
              for (const opp of [opp1, opp2]) {
                if (knowledge.sacrificeFlags[opp] && knowledge.sacrificeFlags[opp].has(newEnd) &&
                    knowledge.avoidanceStrength && knowledge.avoidanceStrength(opp, newEnd) > 0) {
                  ss += 4;
                }
              }
            }

            // Partner support with inference confidence
            const pStr = knowledge.inferStrength(partner);
            let pAff = 0;
            const endNums = newEnd === otherEnd ? [newEnd] : [newEnd, otherEnd];
            for (const endNum of endNums) {
              const played = pStr[endNum] || 0;
              if (played > 0) {
                const remaining = knowledge.remainingWithNumber(endNum);
                let weHold = 0;
                for (const x of tilesByNum[endNum]) { if (x.id !== tile.id) weHold++; }
                const tileHasEnd = (tile.left === endNum || tile.right === endNum) ? 1 : 0;
                const unknownWithEnd = Math.max(0, remaining - weHold - tileHasEnd);
                const tUnk = 28 - knowledge.played.size - hand.length;
                const othersCouldHave = tUnk > 4 ? Math.round(unknownWithEnd * (tUnk - 4) / tUnk) : 0;
                if (othersCouldHave > 0 && !knowledge.cantHave[partner].has(endNum)) pAff += played;
              }
            }
            if (pAff > 0) ss += pAff * 8;
            if (knowledge.cantHave[partner].has(newEnd)) ss -= 10;

            // Partner forward modeling
            const partnerVoidNew = knowledge.cantHave[partner].has(newEnd);
            const partnerVoidOther = knowledge.cantHave[partner].has(otherEnd);
            if (partnerVoidNew && partnerVoidOther) {
              ss -= 20;
            } else if (!partnerVoidNew && !partnerVoidOther) {
              const myIds = new Set(hand.map(h => h.id));
              myIds.add(tile.id);
              let pTilesNew = 0, pTilesOther = 0;
              for (const pt of ALL_TILES) {
                if (knowledge.played.has(pt.id) || myIds.has(pt.id)) continue;
                if (knowledge.cantHave[partner].has(pt.left) || knowledge.cantHave[partner].has(pt.right)) continue;
                if (pt.left === newEnd || pt.right === newEnd) pTilesNew++;
                if (pt.left === otherEnd || pt.right === otherEnd) pTilesOther++;
              }
              if (pTilesNew >= 3 && pTilesOther >= 3) ss += 8;
              else if (pTilesNew === 0 && !partnerVoidNew) ss -= 8;
            }

            // Pip weight — phase-dependent
            const tilePips = tile.left + tile.right;
            ss += Math.round(tilePips * 2 * deadMul);

            // Play doubles early
            if (tile.left === tile.right) ss += 12;

            // Isolated double setup
            for (const h of hand) {
              if (h.id === tile.id || h.left !== h.right) continue;
              let sup = 0;
              for (const x of tilesByNum[h.left]) { if (x.id !== h.id && x.id !== tile.id) sup++; }
              if (sup === 0 && (newEnd === h.left || otherEnd === h.left)) ss += 20;
            }

            // Board counting with gradient
            const remNew = knowledge.remainingWithNumber(newEnd);
            let weHoldNew = 0;
            for (const x of tilesByNum[newEnd]) { if (x.id !== tile.id) weHoldNew++; }
            const unknownWithNew = Math.max(0, remNew - weHoldNew - 1);
            const totalUnknown = 28 - knowledge.played.size - hand.length;
            const oppCouldHaveNew = totalUnknown > 4 ? Math.round(unknownWithNew * (totalUnknown - 4) / totalUnknown) : 0;
            ss += (2 - oppCouldHaveNew) * AI_WEIGHTS.boardCountGradient;

            // Dead number detection — POST-PLAY remaining counts
            const remNewAfterPlay = knowledge.remainingWithNumber(newEnd) - 1;
            const tileHasOtherEnd = (tile.left === otherEnd || tile.right === otherEnd) ? 1 : 0;
            const remOtherAfterPlay = knowledge.remainingWithNumber(otherEnd) - tileHasOtherEnd;
            const deadNew = remNewAfterPlay === 0;
            const deadOther = remOtherAfterPlay === 0;
            if (deadNew && !deadOther) {
              ss -= Math.round(AI_WEIGHTS.deadEndPenalty * deadMul);
            } else if (deadNew && deadOther) {
              const myPipsLock = hand.reduce((s, h) => s + (h.id === tile.id ? 0 : h.left + h.right), 0);
              const myIdsLock = new Set(hand.map(h => h.id));
              let unseenPipLock = 0, unseenCntLock = 0;
              for (const t of ALL_TILES) {
                if (knowledge.played.has(t.id) || myIdsLock.has(t.id)) continue;
                unseenPipLock += t.left + t.right;
                unseenCntLock++;
              }
              const avgPipLock = unseenCntLock > 0 ? unseenPipLock / unseenCntLock : 5;
              const estPartPips = Math.round(Math.max(0, 6 - knowledge.playsBy[partner].length) * avgPipLock);
              const estOpp1T = Math.max(0, 6 - knowledge.playsBy[opp1].length);
              const estOpp2T = Math.max(0, 6 - knowledge.playsBy[opp2].length);
              const estOppPips = Math.round((estOpp1T + estOpp2T) * avgPipLock);
              const myTeamBest = Math.min(myPipsLock, estPartPips);
              const oppTeamBest = Math.min(Math.round(estOpp1T * avgPipLock), Math.round(estOpp2T * avgPipLock));
              if (myTeamBest <= oppTeamBest - 2) {
                ss += Math.round(AI_WEIGHTS.lockFavorable * deadMul);
              } else {
                ss -= Math.round(AI_WEIGHTS.lockUnfavorable * deadMul);
              }
            }

            // Chicote detection
            if (!deadNew && remNewAfterPlay === 1 && newEnd !== otherEnd) {
              const chic = knowledge.chicoteFor ? knowledge.chicoteFor(newEnd, hand) : null;
              if (chic && chic.holder === 'self') {
                ss += Math.round(AI_WEIGHTS.chicoteSelf * deadMul);
                if (knowledge.cantHave[opp1].has(otherEnd) || knowledge.cantHave[opp2].has(otherEnd)) ss += 12;
              } else if (chic && chic.holder === partner) {
                ss += Math.round(AI_WEIGHTS.chicotePartner * deadMul);
              } else if (chic && typeof chic.holder === 'number' && chic.holder !== partner) {
                ss -= Math.round(AI_WEIGHTS.chicoteOpponent * deadMul);
              } else if (chic && chic.holder === 'dorme') {
                ss -= AI_WEIGHTS.chicoteDorme;
              } else if (chic && chic.holder === 'unknown') {
                const oppCount = chic.candidates.filter(c => c !== partner && c !== player).length;
                const oppProb = oppCount / Math.max(chic.candidates.length, 1);
                ss -= Math.round(15 * oppProb);
              } else if (chic) {
                ss -= 12;
              }
            }

            // Probabilistic dead number
            if (!deadNew && remNewAfterPlay > 0 && knowledge.isProbablyDead && knowledge.isProbablyDead(newEnd, player)) {
              let myEndTiles = 0;
              for (const h of hand) {
                if (h.id === tile.id) continue;
                if (h.left === newEnd || h.right === newEnd) myEndTiles++;
              }
              if (myEndTiles > 0) {
                ss += Math.round(AI_WEIGHTS.captiveEndBonus * deadMul);
              } else {
                ss -= Math.round(AI_WEIGHTS.probDeadPenalty * deadMul);
              }
            }

            // Trancar risk/reward
            const remNewT = knowledge.remainingWithNumber(newEnd);
            const remOtherT = knowledge.remainingWithNumber(otherEnd);
            if (remNewT + remOtherT <= 3 && bLen >= 8 && hand.length >= 2) {
              const myPips = hand.reduce((s, h) => s + (h.id === tile.id ? 0 : h.left + h.right), 0);
              const myIds = new Set(hand.map(h => h.id));
              let unseenPipSum = 0, unseenCount = 0;
              for (const t of ALL_TILES) {
                if (knowledge.played.has(t.id) || myIds.has(t.id)) continue;
                unseenPipSum += t.left + t.right;
                unseenCount++;
              }
              const avgPipPerTile = unseenCount > 0 ? unseenPipSum / unseenCount : 5;
              const estOpp1Tiles = Math.max(0, 6 - knowledge.playsBy[opp1].length);
              const estOpp2Tiles = Math.max(0, 6 - knowledge.playsBy[opp2].length);
              const estPartnerTiles = Math.max(0, 6 - knowledge.playsBy[partner].length);
              const estPartnerPips = Math.round(estPartnerTiles * avgPipPerTile);
              const myTeamMin = Math.min(myPips, estPartnerPips);
              const oppTeamMin = Math.min(Math.round(estOpp1Tiles * avgPipPerTile), Math.round(estOpp2Tiles * avgPipPerTile));
              if (myTeamMin < oppTeamMin - 3) {
                ss += AI_WEIGHTS.lockApproachGood;
              } else if (myTeamMin > oppTeamMin + 5) {
                ss -= AI_WEIGHTS.lockApproachBad;
              }
            }

            // Suit exhaustion
            const tNums = tile.left === tile.right ? [tile.left] : [tile.left, tile.right];
            for (const n of tNums) {
              const remAfter = knowledge.remainingWithNumber(n) - 1;
              if (remAfter >= 1 && remAfter <= 3) {
                let weStillHold = 0;
                for (const h of hand) {
                  if (h.id === tile.id) continue;
                  if (h.left === n || h.right === n) weStillHold++;
                }
                if (weStillHold > 0 && weStillHold >= remAfter) {
                  ss += 8 + (3 - remAfter) * 5;
                }
              }
            }

            // True monopoly
            for (const endN of [newEnd, otherEnd]) {
              const remEnd = knowledge.remainingWithNumber(endN);
              if (remEnd >= 1 && remEnd <= 3) {
                let weHoldAll = 0;
                for (const h of hand) {
                  if (h.id === tile.id) continue;
                  if (h.left === endN || h.right === endN) weHoldAll++;
                }
                if (weHoldAll === remEnd) {
                  ss += Math.round(AI_WEIGHTS.monopolyBonus * deadMul);
                }
              }
            }

            // Near-monopoly
            for (const endN of [newEnd, otherEnd]) {
              const remEnd = knowledge.remainingWithNumber(endN);
              if (remEnd >= 2 && remEnd <= 4) {
                let weHoldEnd = 0;
                for (const h of hand) {
                  if (h.id === tile.id) continue;
                  if (h.left === endN || h.right === endN) weHoldEnd++;
                }
                if (weHoldEnd >= 2 && weHoldEnd >= remEnd - 1) {
                  ss += Math.round((weHoldEnd / remEnd) * 12 * deadMul);
                }
              }
            }

            // Information hiding — opening only
            if (bLen >= 1 && bLen <= 6 && hand.length >= 4) {
              const maxSC = Math.max(...suitCount);
              if (suitCount[newEnd] >= 3 && suitCount[newEnd] === maxSC) ss -= 8;
              else if (suitCount[newEnd] <= 1 && newEnd !== otherEnd) ss += 5;
            }

            // Partner close to winning
            const estPHand = Math.max(0, 6 - knowledge.playsBy[partner].length);
            if (estPHand <= 2 && estPHand > 0) {
              const partStr = knowledge.inferStrength(partner);
              if (estPHand === 1) {
                if (partStr[newEnd] >= 2) ss += 25;
                else if (partStr[otherEnd] >= 2) ss += 10;
                else if (partStr[newEnd] >= 1 || partStr[otherEnd] >= 1) ss += 15;
              } else {
                if (partStr[newEnd] >= 1 || partStr[otherEnd] >= 1) ss += 15;
              }
            }

            // Match score awareness
            const ms = matchScores || [0, 0];
            const myTeam = player % 2;
            const ourScore = ms[myTeam], oppScore = ms[1 - myTeam];
            if (ourScore >= 5) {
              ss += Math.round(tilePips * 1.5);
              if (myCount >= 2) ss += 8;
            } else if (oppScore >= 5 && ourScore < 4) {
              if (tile.left === tile.right) ss += 8;
            }

            // Point denial
            for (const opp of [opp1, opp2]) {
              const estOppHand = Math.max(0, 6 - knowledge.playsBy[opp].length);
              if (estOppHand === 1) {
                const oppStr = knowledge.inferStrength(opp);
                for (const eN of [newEnd, otherEnd]) {
                  if (oppStr[eN] >= 2) {
                    const wasThere = (eN === lE || eN === rE);
                    if (!wasThere) {
                      ss += (tile.left === tile.right) ? -15 : -8;
                    }
                  }
                }
              }
            }

            // Closing bonus
            const remainAfter = hand.length - 1;
            if (remainAfter === 0) {
              const w = scoreWin(tile, lE, rE, bLen + 1);
              ss += 200 + w.pts * 30;
            } else if (remainAfter === 1) {
              const lastTile = hand.find(t => t.id !== tile.id);
              if (lastTile) {
                const canPlayLast = (lastTile.left === newEnd || lastTile.right === newEnd ||
                                     lastTile.left === otherEnd || lastTile.right === otherEnd);
                if (canPlayLast) ss += 80;
                else {
                  const allStrand = hand.filter(t => t.id !== tile.id).length === 1 && (() => {
                    const ot = lastTile;
                    const otherPlayable = (ot.left === lE || ot.right === lE || ot.left === rE || ot.right === rE);
                    if (!otherPlayable) return false;
                    for (const oSide of ['left', 'right']) {
                      const oEnd = oSide === 'left' ? lE : rE;
                      if (ot.left !== oEnd && ot.right !== oEnd) continue;
                      const oNewEnd = (ot.left === oEnd) ? ot.right : ot.left;
                      const oOtherEnd = oSide === 'left' ? rE : lE;
                      if (tile.left === oNewEnd || tile.right === oNewEnd || tile.left === oOtherEnd || tile.right === oOtherEnd) return false;
                    }
                    return true;
                  })();
                  ss -= 30;
                }
              }
            } else if (remainAfter === 2) {
              const others = hand.filter(t => t.id !== tile.id);
              const coverCount = others.filter(t =>
                t.left === newEnd || t.right === newEnd || t.left === otherEnd || t.right === otherEnd
              ).length;
              if (coverCount === 2) ss += 25;
              else if (coverCount === 0) ss -= 15;
            }

            options.push({ tile, score: ss, side });
          }
          return options;
        });

        scored.sort((a, b) => b.score - a.score);

        // Two-ply lookahead in endgame
        if (hand.length <= 3 && scored.length >= 2 && bLen >= 10) {
          for (const opt of scored) {
            const { tile: oTile, side: oSide } = opt;
            let oppNewEnd;
            if (oSide === 'left') oppNewEnd = oTile.left === lE ? oTile.right : oTile.left;
            else oppNewEnd = oTile.right === rE ? oTile.left : oTile.right;
            const oppOtherEnd = oSide === 'left' ? rE : lE;

            if (knowledge.possibleTilesFor) {
              const oppPossible = knowledge.possibleTilesFor(opp1);
              const estOpp1Hand = Math.max(0, 6 - knowledge.playsBy[opp1].length);
              if (estOpp1Hand === 1) {
                const oppCanGo = oppPossible.some(t =>
                  t.left === oppNewEnd || t.right === oppNewEnd || t.left === oppOtherEnd || t.right === oppOtherEnd
                );
                if (oppCanGo) opt.score -= 25;
              } else if (estOpp1Hand === 2) {
                const oppPlayable = oppPossible.filter(t =>
                  t.left === oppNewEnd || t.right === oppNewEnd || t.left === oppOtherEnd || t.right === oppOtherEnd
                ).length;
                if (oppPlayable >= 2) opt.score -= 10;
              }
            }
          }
          scored.sort((a, b) => b.score - a.score);
        }

        if (returnAll) return scored;
        return scored[0] || null;
      };

      // === Monte Carlo Rollout System ===

      // Fast rollout AI — lightweight scoring for MC simulations (with dead number detection)
      const fastAI = (hand, lE, rE, bLen, player, knowledge) => {
        const playable = hand.filter(t => bLen === 0 || t.left === lE || t.right === lE || t.left === rE || t.right === rE);
        if (playable.length === 0) return null;
        if (playable.length === 1) return { tile: playable[0], side: null };

        const partner = (player + 2) % 4;
        const opp1 = (player + 1) % 4, opp2 = (player + 3) % 4;
        const sc = new Array(7).fill(0);
        for (const t of hand) { sc[t.left]++; if (t.left !== t.right) sc[t.right]++; }

        if (bLen === 0) {
          let best = null, bestS = -Infinity;
          for (const t of playable) {
            const s = sc[t.left]*10 + sc[t.right]*10 + (t.left===t.right?15:0) + (t.left+t.right)*2;
            if (s > bestS) { bestS = s; best = t; }
          }
          return { tile: best, side: null };
        }

        const deadMul = bLen <= 4 ? 0.4 : (bLen <= 14 ? 1.0 : 1.4);
        let bestTile = null, bestSide = null, bestScore = -Infinity;
        for (const tile of playable) {
          for (const side of ['left', 'right']) {
            const cS = side === 'left' ? (tile.left === lE || tile.right === lE) : (tile.left === rE || tile.right === rE);
            if (!cS) continue;
            let newEnd = side === 'left' ? (tile.left === lE ? tile.right : tile.left) : (tile.right === rE ? tile.left : tile.right);
            const otherEnd = side === 'left' ? rE : lE;
            let ss = 0;
            if (hand.length === 1) return { tile, side };
            let mc = 0;
            for (const h of hand) {
              if (h.id !== tile.id && (h.left === newEnd || h.right === newEnd || h.left === otherEnd || h.right === otherEnd)) mc++;
            }
            ss += mc * 12;
            if (knowledge.cantHave[opp1].has(newEnd) && knowledge.cantHave[opp1].has(otherEnd)) ss += 25;
            if (knowledge.cantHave[opp2].has(newEnd) && knowledge.cantHave[opp2].has(otherEnd)) ss += 18;
            if (knowledge.cantHave[partner].has(newEnd) && knowledge.cantHave[partner].has(otherEnd)) ss -= 15;
            ss += Math.round((tile.left + tile.right) * 2 * deadMul);
            if (tile.left === tile.right) ss += 10;

            // Dead number detection
            {
              const remNew = knowledge.remainingWithNumber(newEnd);
              const remNewAfter = remNew - 1;
              if (remNewAfter === 0 && newEnd !== otherEnd) {
                const tileHasOther = (tile.left === otherEnd || tile.right === otherEnd) ? 1 : 0;
                const remOther = knowledge.remainingWithNumber(otherEnd) - tileHasOther;
                if (remOther > 0) {
                  ss -= 30;
                } else {
                  let myPips = 0;
                  for (const h of hand) { if (h.id !== tile.id) myPips += h.left + h.right; }
                  ss += myPips <= 6 ? 20 : -25;
                }
              }
            }

            // Near-close checks
            if (hand.length === 2) {
              const last = hand.find(t => t.id !== tile.id);
              if (last && (last.left === newEnd || last.right === newEnd || last.left === otherEnd || last.right === otherEnd)) ss += 60;
              else ss -= 30;
            }
            if (hand.length === 3) {
              let coverCount = 0;
              for (const h of hand) {
                if (h.id === tile.id) continue;
                if (h.left === newEnd || h.right === newEnd || h.left === otherEnd || h.right === otherEnd) coverCount++;
              }
              if (coverCount === 2) ss += 20;
              else if (coverCount === 0) ss -= 15;
            }

            if (ss > bestScore) { bestScore = ss; bestTile = tile; bestSide = side; }
          }
        }
        return bestTile ? { tile: bestTile, side: bestSide } : { tile: playable[0], side: null };
      };

      // Resolve a blocked game — returns { team, points, type }
      const resolveBlock = (hands) => {
        const vals = hands.map((h, i) => ({ p: i, pts: h.reduce((s, t) => s + t.left + t.right, 0) }));
        const min = Math.min(...vals.map(v => v.pts));
        const winners = vals.filter(v => v.pts === min);
        if (winners.length > 1 && winners.some(w => w.p % 2 === 0) && winners.some(w => w.p % 2 === 1)) {
          return { team: -1, points: 0, type: 'tie' };
        }
        return { team: winners[0].p % 2, points: 1, type: 'blocked' };
      };

      // Simulate a complete game from a position using fastAI
      const simulateFromPosition = (hands, lE, rE, nextPlayer, knowledge, bLen) => {
        const simHands = hands.map(h => [...h]);
        let simLE = lE, simRE = rE, cur = nextPlayer, passCount = 0;
        const simK = {
          cantHave: knowledge.cantHave.map(s => new Set(s)),
          played: new Set(knowledge.played),
          playsBy: knowledge.playsBy.map(a => [...a]),
          _rc: [...Array(7)].map((_, n) => knowledge.remainingWithNumber(n)),
          _sc: [null, null, null, null],
          remainingWithNumber(n) { return this._rc[n]; },
          inferStrength(p) {
            if (this._sc[p]) return this._sc[p];
            const s = [0,0,0,0,0,0,0];
            for (const t of this.playsBy[p]) { s[t.left]++; if (t.left !== t.right) s[t.right]++; }
            this._sc[p] = s; return s;
          },
          recordPlay(p, t) {
            if (!this.played.has(t.id)) {
              this.played.add(t.id);
              this._rc[t.left]--;
              if (t.left !== t.right) this._rc[t.right]--;
            }
            this.playsBy[p].push(t);
            this._sc[p] = null;
          },
          recordPass(p, le, re) {
            this.cantHave[p].add(le);
            this.cantHave[p].add(re);
          }
        };

        for (let move = 0; move < 100; move++) {
          const decision = fastAI(simHands[cur], simLE, simRE, bLen, cur, simK);
          if (decision) {
            passCount = 0;
            const { tile, side: prefSide } = decision;
            let side = prefSide;
            if (!side) side = (tile.left === simLE || tile.right === simLE) ? 'left' : 'right';
            simHands[cur] = simHands[cur].filter(t => t.id !== tile.id);
            simK.recordPlay(cur, tile);
            const prevLE = simLE, prevRE = simRE;
            if (bLen === 0) { simLE = tile.left; simRE = tile.right; }
            else if (side === 'left') simLE = tile.left === simLE ? tile.right : tile.left;
            else simRE = tile.right === simRE ? tile.left : tile.right;
            bLen++;
            if (simHands[cur].length === 0) {
              const isD = tile.left === tile.right;
              const wasBoth = couldPlayOnBothEnds(tile, prevLE, prevRE);
              const pts = isD && wasBoth ? 4 : isD ? 2 : wasBoth ? 3 : 1;
              return { winnerTeam: cur % 2, points: pts };
            }
            cur = (cur + 1) % 4;
          } else {
            passCount++;
            simK.recordPass(cur, simLE, simRE);
            if (passCount >= 4) return resolveBlock(simHands);
            cur = (cur + 1) % 4;
          }
        }
        return { winnerTeam: -1, points: 0, type: 'tie' };
      };

      // Helper: can player hold this tile given knowledge constraints?
      const canPlayerHold = (p, t, knowledge) => {
        if (knowledge.cantHave[p].has(t.left)) return false;
        if (knowledge.cantHave[p].has(t.right)) return false;
        return true;
      };

      // Soft-belief weighted shuffle: bias tile assignment toward play-pattern plausibility
      const softWeightedShuffle = (avail, player, knowledge) => {
        if (knowledge.played.size <= 4 || avail.length <= 1) {
          for (let i = avail.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [avail[i], avail[j]] = [avail[j], avail[i]];
          }
          return;
        }
        const str = knowledge.inferStrength(player);
        const hasSignal = str.some(s => s >= 2);
        if (!hasSignal) {
          for (let i = avail.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [avail[i], avail[j]] = [avail[j], avail[i]];
          }
          return;
        }
        for (let i = avail.length - 1; i > 0; i--) {
          let wSum = 0;
          for (let j = 0; j <= i; j++) {
            const t = avail[j];
            const w = 1 + 0.6 * ((str[t.left] || 0) + (str[t.right] || 0));
            avail[j]._sw = w;
            wSum += w;
          }
          let r = Math.random() * wSum;
          let pick = 0;
          for (; pick < i; pick++) {
            r -= avail[pick]._sw;
            if (r <= 0) break;
          }
          if (pick !== i) { const tmp = avail[i]; avail[i] = avail[pick]; avail[pick] = tmp; }
        }
        for (const t of avail) delete t._sw;
      };

      // Generate a consistent deal using constraint propagation (most-constrained-first)
      const generateConsistentDeal = (myHand, handSizes, knowledge, mySlot) => {
        const myIds = new Set(myHand.map(t => t.id));
        const pool = [];
        for (const t of ALL_TILES) {
          if (knowledge.played.has(t.id) || myIds.has(t.id)) continue;
          pool.push(t);
        }
        const players = [0, 1, 2, 3].filter(p => p !== mySlot);

        // Pre-compute per-player eligible tiles
        const eligible = {};
        for (const p of players) {
          eligible[p] = pool.filter(t => canPlayerHold(p, t, knowledge));
        }

        // Sort players by slack (eligible - needed) ascending = most constrained first
        const sortedPlayers = [...players].sort((a, b) =>
          (eligible[a].length - handSizes[a]) - (eligible[b].length - handSizes[b])
        );

        for (let attempt = 0; attempt < 100; attempt++) {
          const hands = [[], [], [], []];
          hands[mySlot] = myHand;
          const assigned = new Set();
          let valid = true;

          for (const p of sortedPlayers) {
            const avail = eligible[p].filter(t => !assigned.has(t.id));
            if (avail.length < handSizes[p]) { valid = false; break; }
            softWeightedShuffle(avail, p, knowledge);
            for (let i = 0; i < handSizes[p]; i++) {
              hands[p].push(avail[i]);
              assigned.add(avail[i].id);
            }
            for (const t of hands[p]) {
              if (!canPlayerHold(p, t, knowledge)) { valid = false; break; }
            }
            if (!valid) break;
          }
          if (valid) return hands;
        }

        // Fallback: rejection sampling
        const shuffled = new Array(pool.length);
        for (let attempt = 0; attempt < 200; attempt++) {
          for (let i = 0; i < pool.length; i++) shuffled[i] = pool[i];
          for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
          }
          const hands = [[], [], [], []];
          hands[mySlot] = myHand;
          let idx = 0, valid = true;
          for (const p of players) {
            for (let i = 0; i < handSizes[p]; i++) {
              if (idx >= shuffled.length) { valid = false; break; }
              const t = shuffled[idx++];
              if (!canPlayerHold(p, t, knowledge)) { valid = false; break; }
              hands[p].push(t);
            }
            if (!valid) break;
          }
          if (valid) return hands;
        }

        // Last resort: partial deal
        const hands = [[], [], [], []];
        hands[mySlot] = myHand;
        const assigned = new Set();
        for (const p of sortedPlayers) {
          const avail = eligible[p].filter(t => !assigned.has(t.id));
          for (let i = avail.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [avail[i], avail[j]] = [avail[j], avail[i]];
          }
          for (let i = 0; i < Math.min(handSizes[p], avail.length); i++) {
            hands[p].push(avail[i]);
            assigned.add(avail[i].id);
          }
        }
        return hands;
      };

      // Monte Carlo evaluation — run N simulations per move, pick best
      const MC_SIMS = 80;
      const monteCarloEval = (hand, lE, rE, bLen, player, knowledge) => {
        const canPlayMC = (t) => bLen === 0 || t.left === lE || t.right === lE || t.left === rE || t.right === rE;
        const playable = hand.filter(canPlayMC);
        if (playable.length === 0) return null;
        if (playable.length === 1) {
          const t = playable[0];
          const side = bLen === 0 ? null : (t.left === lE || t.right === lE) ? 'left' : 'right';
          return { tile: t, side };
        }

        const myTeam = player % 2;
        const handSizes = [0, 0, 0, 0];
        // Estimate hand sizes from knowledge
        for (let p = 0; p < 4; p++) {
          if (p === player) { handSizes[p] = hand.length; continue; }
          handSizes[p] = Math.max(0, 6 - knowledge.playsBy[p].length);
        }

        // Build all move options
        const options = [];
        for (const tile of playable) {
          for (const side of ['left', 'right']) {
            if (bLen === 0) {
              options.push({ tile, side: null, wins: 0, totalPts: 0, sims: 0 });
              break;
            }
            const cS = side === 'left' ? (tile.left === lE || tile.right === lE) : (tile.left === rE || tile.right === rE);
            if (!cS) continue;
            options.push({ tile, side, wins: 0, totalPts: 0, sims: 0 });
          }
        }

        // Run simulations
        for (let sim = 0; sim < MC_SIMS; sim++) {
          for (const opt of options) {
            // Apply this move
            const newHand = hand.filter(t => t.id !== opt.tile.id);
            let newLE = lE, newRE = rE, newBLen = bLen;
            if (bLen === 0) { newLE = opt.tile.left; newRE = opt.tile.right; }
            else if (opt.side === 'left') newLE = opt.tile.left === lE ? opt.tile.right : opt.tile.left;
            else newRE = opt.tile.right === rE ? opt.tile.left : opt.tile.right;
            newBLen++;

            // Instant win
            if (newHand.length === 0) {
              const isD = opt.tile.left === opt.tile.right;
              const wasBoth = couldPlayOnBothEnds(opt.tile, lE, rE);
              const pts = isD && wasBoth ? 4 : isD ? 2 : wasBoth ? 3 : 1;
              opt.wins += 1;
              opt.totalPts += pts;
              opt.sims++;
              continue;
            }

            // Clone knowledge and record our play
            const simK = knowledge.clone ? knowledge.clone() : (() => {
              const rc = [...Array(7)].map((_, n) => knowledge.remainingWithNumber(n));
              rc[opt.tile.left]--;
              if (opt.tile.left !== opt.tile.right) rc[opt.tile.right]--;
              const playsBy = knowledge.playsBy.map(a => [...a]);
              playsBy[player].push(opt.tile);
              return {
                cantHave: knowledge.cantHave.map(s => new Set(s)),
                played: new Set([...knowledge.played, opt.tile.id]),
                playsBy,
                remainingWithNumber: (n) => rc[n],
                inferStrength: (p) => {
                  const s = [0,0,0,0,0,0,0];
                  for (const t of playsBy[p]) { s[t.left]++; if (t.left !== t.right) s[t.right]++; }
                  return s;
                }
              };
            })();
            if (simK.recordPlay) simK.recordPlay(player, opt.tile);
            else {
              simK.played.add(opt.tile.id);
              simK.playsBy[player].push(opt.tile);
            }

            // Generate consistent deal and simulate
            const newHandSizes = [...handSizes];
            newHandSizes[player] = newHand.length;
            const deal = generateConsistentDeal(newHand, newHandSizes, simK, player);
            const result = simulateFromPosition(deal, newLE, newRE, (player + 1) % 4, simK, newBLen);

            if (result.winnerTeam === myTeam) {
              opt.wins++;
              opt.totalPts += result.points;
            } else if (result.winnerTeam >= 0) {
              opt.totalPts -= result.points;
            }
            opt.sims++;
          }
        }

        // Pick best: highest win rate, break ties by expected points
        let best = null, bestWinRate = -1, bestPts = -Infinity;
        const tileResults = new Map();
        for (const opt of options) {
          if (opt.sims === 0) continue;
          const wr = opt.wins / opt.sims;
          const ep = opt.totalPts / opt.sims;
          const key = opt.tile.id + '|' + opt.side;
          if (!tileResults.has(opt.tile.id) || wr > tileResults.get(opt.tile.id).wr || (wr === tileResults.get(opt.tile.id).wr && ep > tileResults.get(opt.tile.id).ep)) {
            tileResults.set(opt.tile.id, { tile: opt.tile, side: opt.side, wr, ep });
          }
        }
        for (const [, r] of tileResults) {
          if (r.wr > bestWinRate || (r.wr === bestWinRate && r.ep > bestPts)) {
            bestWinRate = r.wr;
            bestPts = r.ep;
            best = { tile: r.tile, side: r.side };
          }
        }
        return best;
      };

      // === TILE INDEX LOOKUP ===
      const TILE_INDEX = {};
      ALL_TILES.forEach((t, i) => { TILE_INDEX[t.id] = i; });

      // === BITMASK ENDGAME LOOKUP TABLES ===
      const _EG_TILE_LEFT   = new Int8Array(28);
      const _EG_TILE_RIGHT  = new Int8Array(28);
      const _EG_TILE_PIPS   = new Int8Array(28);
      const _EG_TILE_DOUBLE = new Uint8Array(28);
      const _EG_TILES_WITH_PIP = new Int32Array(7);
      (() => {
        for (let i = 0; i < 28; i++) {
          const t = ALL_TILES[i];
          _EG_TILE_LEFT[i] = t.left;
          _EG_TILE_RIGHT[i] = t.right;
          _EG_TILE_PIPS[i] = t.left + t.right;
          _EG_TILE_DOUBLE[i] = (t.left === t.right) ? 1 : 0;
        }
        for (let n = 0; n < 7; n++) {
          let mask = 0;
          for (let i = 0; i < 28; i++) {
            if (_EG_TILE_LEFT[i] === n || _EG_TILE_RIGHT[i] === n) mask |= (1 << i);
          }
          _EG_TILES_WITH_PIP[n] = mask;
        }
      })();

      // === SplitMix64 PRNG for Zobrist hashing ===
      class SplitMix64 {
        constructor(seed) {
          this.s0 = seed >>> 0;
          this.s1 = (seed / 0x100000000) >>> 0;
        }
        _add64(aLo, aHi, bLo, bHi) {
          const lo = (aLo + bLo) >>> 0;
          const hi = (aHi + bHi + (lo < aLo ? 1 : 0)) >>> 0;
          return [lo, hi];
        }
        _xorshift(lo, hi, bits) {
          if (bits < 32) {
            const rLo = ((lo >>> bits) | (hi << (32 - bits))) >>> 0;
            const rHi = (hi >>> bits) >>> 0;
            return [(lo ^ rLo) >>> 0, (hi ^ rHi) >>> 0];
          }
          const b = bits - 32;
          return [(lo ^ (hi >>> b)) >>> 0, hi];
        }
        _mul64(aLo, aHi, bLo, bHi) {
          const al = aLo & 0xFFFF, ah = aLo >>> 16;
          const bl = bLo & 0xFFFF, bh = bLo >>> 16;
          let lo = al * bl;
          let mid = ah * bl + (lo >>> 16);
          mid += al * bh;
          let hi = (mid >>> 16) + ah * bh;
          lo = ((mid & 0xFFFF) << 16) | (lo & 0xFFFF);
          hi = (hi + aLo * bHi + aHi * bLo) >>> 0;
          return [lo >>> 0, hi >>> 0];
        }
        next() {
          const [lo, hi] = this._add64(this.s0, this.s1, 0x7f4a7c15, 0x9e3779b9);
          this.s0 = lo; this.s1 = hi;
          let [zLo, zHi] = [lo, hi];
          [zLo, zHi] = this._xorshift(zLo, zHi, 30);
          [zLo, zHi] = this._mul64(zLo, zHi, 0x1ce4e5b9, 0xbf58476d);
          [zLo, zHi] = this._xorshift(zLo, zHi, 27);
          [zLo, zHi] = this._mul64(zLo, zHi, 0x133111eb, 0x94d049bb);
          [zLo, zHi] = this._xorshift(zLo, zHi, 31);
          return [zLo, zHi];
        }
        random() { const [lo] = this.next(); return (lo >>> 0) / 0x100000000; }
        randomInt(max) { return Math.floor(this.random() * max); }
      }

      // === ZOBRIST HASH TABLES ===
      const _ZOBRIST = (() => {
        const rng = new SplitMix64(0xDEADBEEF);
        const tp = [];
        for (let t = 0; t < 28; t++) {
          tp[t] = [];
          for (let p = 0; p < 4; p++) tp[t][p] = rng.randomInt(0x7FFFFFFF);
        }
        const le = [], re = [];
        for (let n = 0; n < 7; n++) { le[n] = rng.randomInt(0x7FFFFFFF); re[n] = rng.randomInt(0x7FFFFFFF); }
        const np = [];
        for (let p = 0; p < 4; p++) np[p] = rng.randomInt(0x7FFFFFFF);
        const pc = [];
        for (let c = 0; c < 5; c++) pc[c] = rng.randomInt(0x7FFFFFFF);
        const bl = [];
        for (let b = 0; b < 25; b++) bl[b] = rng.randomInt(0x7FFFFFFF);
        return { tp, le, re, np, pc, bl };
      })();

      // === MATCH EQUITY TABLE ===
      const MATCH_TARGET = 6;
      const POINT_DIST = [
        { pts: 1, prob: 0.70 },
        { pts: 2, prob: 0.16 },
        { pts: 3, prob: 0.10 },
        { pts: 4, prob: 0.04 }
      ];
      const DOB_VALUES = [1, 2, 4, 8];
      const ME3D = (() => {
        const T = MATCH_TARGET;
        const S = T + 5;
        const me = Array.from({length: S}, () =>
          Array.from({length: S}, () => new Float64Array(DOB_VALUES.length))
        );
        for (let d = 0; d < DOB_VALUES.length; d++) {
          for (let s2 = 0; s2 < S; s2++) {
            for (let s1 = T; s1 < S; s1++) me[s1][s2][d] = s2 >= T ? 0.5 : 1.0;
          }
          for (let s1 = 0; s1 < T; s1++) {
            for (let s2 = T; s2 < S; s2++) me[s1][s2][d] = 0.0;
          }
        }
        const TIE_PROB = 0.03;
        for (let s1 = T - 1; s1 >= 0; s1--) {
          for (let s2 = T - 1; s2 >= 0; s2--) {
            for (let d = DOB_VALUES.length - 1; d >= 0; d--) {
              const dob = DOB_VALUES[d];
              let decisive = 0;
              for (const { pts: basePts, prob } of POINT_DIST) {
                const pts = basePts * dob;
                const s1w = Math.min(s1 + pts, T + 4);
                const s2w = Math.min(s2 + pts, T + 4);
                decisive += 0.5 * prob * me[s1w][s2][0] + 0.5 * prob * me[s1][s2w][0];
              }
              const nextDobIdx = Math.min(d + 1, DOB_VALUES.length - 1);
              if (d === DOB_VALUES.length - 1) {
                me[s1][s2][d] = decisive;
              } else {
                me[s1][s2][d] = decisive * (1 - TIE_PROB) + TIE_PROB * me[s1][s2][nextDobIdx];
              }
            }
          }
        }
        return me;
      })();

      const getMatchEquity3D = (s1, s2, dobMultiplier) => {
        const dIdx = DOB_VALUES.indexOf(dobMultiplier);
        const d = dIdx >= 0 ? dIdx : 0;
        return ME3D[Math.min(s1, MATCH_TARGET + 4)][Math.min(s2, MATCH_TARGET + 4)][d];
      };

      // Convert rollout result to match equity delta
      const _rolloutToMEReward = (winnerTeam, points, myTeam, matchScores, dobMultiplier) => {
        if (winnerTeam < 0) return 0;
        const myScore = matchScores[myTeam], oppScore = matchScores[1 - myTeam];
        const dob = dobMultiplier || 1;
        const currentME = getMatchEquity3D(myScore, oppScore, dob);
        const pts = points * dob;
        let newME;
        if (winnerTeam === myTeam) {
          newME = getMatchEquity3D(Math.min(myScore + pts, MATCH_TARGET + 4), oppScore, 1);
        } else {
          newME = getMatchEquity3D(myScore, Math.min(oppScore + pts, MATCH_TARGET + 4), 1);
        }
        return newME - currentME;
      };

      // === BITMASK ENDGAME SOLVER ===
      const ENDGAME_BUDGET_MS = 500;
      const ENDGAME_SAMPLE_COUNT = 100;

      const _egResolveBlock = (handBits) => {
        const pts = [0, 0, 0, 0];
        for (let p = 0; p < 4; p++) {
          let mask = handBits[p];
          while (mask) {
            const bit = mask & (-mask);
            pts[p] += _EG_TILE_PIPS[31 - Math.clz32(bit)];
            mask ^= bit;
          }
        }
        const min = Math.min(pts[0], pts[1], pts[2], pts[3]);
        const winners = [];
        for (let p = 0; p < 4; p++) { if (pts[p] === min) winners.push(p); }
        if (winners.length > 1 && winners.some(w => (w % 2) === 0) && winners.some(w => (w % 2) === 1)) {
          return { type: 'tie', points: 0 };
        }
        return { type: 'blocked', team: winners[0] % 2, points: 1 };
      };

      const _egScoreWin = (tileIdx, prevLE, prevRE, boardLen) => {
        const isD = _EG_TILE_DOUBLE[tileIdx];
        let wasBoth = false;
        if (boardLen > 1 && prevLE >= 0 && prevRE >= 0) {
          const tL = _EG_TILE_LEFT[tileIdx], tR = _EG_TILE_RIGHT[tileIdx];
          if (prevLE === prevRE) {
            wasBoth = (tL === prevLE && tR === prevLE);
          } else {
            wasBoth = (tL === prevLE || tR === prevLE) && (tL === prevRE || tR === prevRE);
          }
        }
        if (isD && wasBoth) return 4;
        if (isD) return 2;
        if (wasBoth) return 3;
        return 1;
      };

      const _egHashInit = (handBits, lE, rE, bLen, nextPlayer, passCount) => {
        let h = 0;
        for (let p = 0; p < 4; p++) {
          let mask = handBits[p];
          while (mask) {
            const bit = mask & (-mask);
            h ^= _ZOBRIST.tp[31 - Math.clz32(bit)][p];
            mask ^= bit;
          }
        }
        h ^= _ZOBRIST.bl[Math.min(bLen, 24)];
        if (bLen > 0) { h ^= _ZOBRIST.le[lE]; h ^= _ZOBRIST.re[rE]; }
        h ^= _ZOBRIST.np[nextPlayer];
        h ^= _ZOBRIST.pc[Math.min(passCount, 4)];
        return h;
      };

      const _egMoveScore = (tileIdx, handAfter, lE, rE, bLen, ttBestTile) => {
        let s = 0;
        if (handAfter === 0) s += 10000;
        if (tileIdx === ttBestTile) s += 5000;
        if (_EG_TILE_DOUBLE[tileIdx]) s += 100;
        s += _EG_TILE_PIPS[tileIdx] * 3;
        if (bLen > 0) {
          const tL = _EG_TILE_LEFT[tileIdx], tR = _EG_TILE_RIGHT[tileIdx];
          if ((tL === lE || tR === lE) && (tL === rE || tR === rE)) s += 50;
        }
        return s;
      };

      const _endgameMinimaxBit = (handBits, lE, rE, bLen, nextPlayer, passCount, myTeam, alpha, beta, deadline, tt, hash, matchScores, dobMul) => {
        if (Date.now() > deadline) return 0;
        if (passCount >= 4) {
          const block = _egResolveBlock(handBits);
          return block.type === 'tie' ? 0 : _rolloutToMEReward(block.team, block.points, myTeam, matchScores, dobMul);
        }

        let ttBestTile = -1;
        const ttEntry = tt.get(hash);
        if (ttEntry) {
          if (ttEntry.f === 0) return ttEntry.v;
          if (ttEntry.f === 1 && ttEntry.v >= beta) return ttEntry.v;
          if (ttEntry.f === 2 && ttEntry.v <= alpha) return ttEntry.v;
          if (ttEntry.f === 1 && ttEntry.v > alpha) alpha = ttEntry.v;
          if (ttEntry.f === 2 && ttEntry.v < beta) beta = ttEntry.v;
          if (ttEntry.m >= 0) ttBestTile = ttEntry.m;
        }
        const origAlpha = alpha;
        const origBeta = beta;

        const hand = handBits[nextPlayer];
        const playableMask = (bLen === 0) ? hand : (hand & (_EG_TILES_WITH_PIP[lE] | _EG_TILES_WITH_PIP[rE]));

        if (playableMask === 0) {
          const newNext = (nextPlayer + 1) % 4;
          const newPass = passCount + 1;
          let ph = hash;
          ph ^= _ZOBRIST.np[nextPlayer]; ph ^= _ZOBRIST.np[newNext];
          ph ^= _ZOBRIST.pc[Math.min(passCount, 4)]; ph ^= _ZOBRIST.pc[Math.min(newPass, 4)];
          const val = _endgameMinimaxBit(handBits, lE, rE, bLen, newNext, newPass, myTeam, alpha, beta, deadline, tt, ph, matchScores, dobMul);
          tt.set(hash, { v: val, f: 0, m: -1 });
          return val;
        }

        const moves = [];
        let rem = playableMask;
        while (rem) {
          const bit = rem & (-rem);
          const tileIdx = 31 - Math.clz32(bit);
          rem ^= bit;
          const tL = _EG_TILE_LEFT[tileIdx], tR = _EG_TILE_RIGHT[tileIdx];
          const sides = [];
          if (bLen === 0) {
            sides.push(0);
          } else {
            if (tL === lE || tR === lE) sides.push(1);
            if ((tL === rE || tR === rE) && lE !== rE) sides.push(2);
            if (sides.length === 0 && (tL === rE || tR === rE)) sides.push(2);
          }
          for (let si = 0; si < sides.length; si++) {
            const handAfter = hand ^ bit;
            moves.push({ bit, tileIdx, tL, tR, side: sides[si], score: _egMoveScore(tileIdx, handAfter, lE, rE, bLen, ttBestTile) });
          }
        }
        if (moves.length > 1) moves.sort((a, b) => b.score - a.score);

        const isMax = (nextPlayer % 2) === myTeam;
        let bestVal = isMax ? -Infinity : Infinity;
        let bestMove = -1;
        const newNext = (nextPlayer + 1) % 4;

        for (let mi = 0; mi < moves.length; mi++) {
          if (Date.now() > deadline) break;
          const { bit, tileIdx, tL, tR, side } = moves[mi];
          let newLE = lE, newRE = rE;
          if (side === 0) { newLE = tL; newRE = tR; }
          else if (side === 1) { newLE = (tL === lE) ? tR : tL; }
          else { newRE = (tR === rE) ? tL : tR; }

          handBits[nextPlayer] ^= bit;
          let val;
          if (handBits[nextPlayer] === 0) {
            const pts = _egScoreWin(tileIdx, lE, rE, bLen + 1);
            val = _rolloutToMEReward(nextPlayer % 2, pts, myTeam, matchScores, dobMul);
          } else {
            let mh = hash;
            mh ^= _ZOBRIST.tp[tileIdx][nextPlayer];
            mh ^= _ZOBRIST.bl[Math.min(bLen, 24)];
            if (bLen > 0) { mh ^= _ZOBRIST.le[lE]; mh ^= _ZOBRIST.re[rE]; }
            mh ^= _ZOBRIST.np[nextPlayer]; mh ^= _ZOBRIST.pc[Math.min(passCount, 4)];
            mh ^= _ZOBRIST.bl[Math.min(bLen + 1, 24)];
            mh ^= _ZOBRIST.le[newLE]; mh ^= _ZOBRIST.re[newRE];
            mh ^= _ZOBRIST.np[newNext]; mh ^= _ZOBRIST.pc[0];
            val = _endgameMinimaxBit(handBits, newLE, newRE, bLen + 1, newNext, 0, myTeam, alpha, beta, deadline, tt, mh, matchScores, dobMul);
          }
          handBits[nextPlayer] ^= bit;

          if (isMax) {
            if (val > bestVal) { bestVal = val; bestMove = tileIdx; }
            if (bestVal > alpha) alpha = bestVal;
          } else {
            if (val < bestVal) { bestVal = val; bestMove = tileIdx; }
            if (bestVal < beta) beta = bestVal;
          }
          if (beta <= alpha) break;
        }

        let flag;
        if (bestVal <= origAlpha) flag = 2;
        else if (bestVal >= origBeta) flag = 1;
        else flag = 0;
        tt.set(hash, { v: bestVal, f: flag, m: bestMove });
        return bestVal;
      };

      // Endgame solver entry point
      const endgameSolve = (hand, lE, rE, bLen, player, knowledge, matchScores, dobMultiplier) => {
        if (bLen === 0) return null;
        const deadline = Date.now() + ENDGAME_BUDGET_MS;
        const myTeam = player % 2;
        const canPlayTile = (t) => t.left === lE || t.right === lE || t.left === rE || t.right === rE;
        const playable = hand.filter(canPlayTile);
        if (playable.length === 0) return null;
        if (playable.length === 1) return { tile: playable[0], side: null };

        // Compute hand sizes from knowledge
        const playedPerPlayer = knowledge.playsBy.map(a => a.length);
        const handSizes = [0, 0, 0, 0];
        for (let p = 0; p < 4; p++) {
          handSizes[p] = p === player ? hand.length : Math.max(1, 6 - playedPerPlayer[p]);
        }

        // Sample consistent deals and solve with minimax
        const deals = [];
        for (let i = 0; i < ENDGAME_SAMPLE_COUNT && Date.now() < deadline; i++) {
          try {
            const deal = generateConsistentDeal(hand, handSizes, knowledge, player);
            deals.push(deal);
          } catch(e) { /* skip failed samples */ }
        }
        if (deals.length === 0) return null;

        // Build move registry
        const moveEntries = [];
        for (const tile of playable) {
          const sides = [];
          if (tile.left === lE || tile.right === lE) sides.push('left');
          if ((tile.left === rE || tile.right === rE) && lE !== rE) sides.push('right');
          if (sides.length === 0 && (tile.left === rE || tile.right === rE)) sides.push('right');
          for (const side of sides) {
            moveEntries.push({ tile, side, totalME: 0, solvedWeight: 0, deals: 0 });
          }
        }

        const ms = matchScores || [0, 0];
        const dm = dobMultiplier || 1;

        for (let d = 0; d < deals.length && Date.now() < deadline; d++) {
          const deal = deals[d];
          const w = 1 / deals.length;
          const tt = new Map();

          const dealBits = new Int32Array(4);
          for (let p = 0; p < 4; p++) {
            let mask = 0;
            const src = (p === player) ? hand : deal[p];
            for (let i = 0; i < src.length; i++) mask |= (1 << TILE_INDEX[src[i].id]);
            dealBits[p] = mask;
          }

          for (let m = 0; m < moveEntries.length; m++) {
            if (Date.now() > deadline) break;
            const entry = moveEntries[m];
            const { tile, side } = entry;
            const tileIdx = TILE_INDEX[tile.id];
            const tileBit = 1 << tileIdx;

            let newLE = lE, newRE = rE;
            if (side === 'left') { newLE = tile.left === lE ? tile.right : tile.left; }
            else { newRE = tile.right === rE ? tile.left : tile.right; }

            const solveBits = new Int32Array(dealBits);
            solveBits[player] ^= tileBit;

            if (solveBits[player] === 0) {
              const pts = _egScoreWin(tileIdx, lE, rE, bLen + 1);
              const reward = _rolloutToMEReward(player % 2, pts, myTeam, ms, dm);
              entry.totalME += w * reward;
              entry.solvedWeight += w;
              entry.deals++;
              continue;
            }

            const initHash = _egHashInit(solveBits, newLE, newRE, bLen + 1, (player + 1) % 4, 0);
            const val = _endgameMinimaxBit(solveBits, newLE, newRE, bLen + 1, (player + 1) % 4, 0, myTeam, -Infinity, Infinity, deadline, tt, initHash, ms, dm);
            entry.totalME += w * val;
            entry.solvedWeight += w;
            entry.deals++;
          }
        }

        // Pick best move per tile
        const tileResults = new Map();
        for (const entry of moveEntries) {
          if (entry.solvedWeight === 0) continue;
          const me = entry.totalME / entry.solvedWeight;
          const result = { tile: entry.tile, side: entry.side, expectedPoints: me * 4 };
          const tileKey = entry.tile.id;
          if (!tileResults.has(tileKey) || result.expectedPoints > tileResults.get(tileKey).expectedPoints) {
            tileResults.set(tileKey, result);
          }
        }

        const results = [...tileResults.values()];
        results.sort((a, b) => b.expectedPoints - a.expectedPoints);
        return results.length > 0 ? results[0] : null;
      };

      // NN helper functions (globals NN_STATE_DIM, NN_NUM_ACTIONS, _nnModel, USE_NN_LEAF_VALUE, _nnLeafStats, loadNeuralModel are above App)
      const _nnMatVec = (W, b, x) => {
        const rows = W.shape[0], cols = W.shape[1];
        const out = new Float32Array(rows);
        for (let r = 0; r < rows; r++) {
          let sum = b.data[r];
          const rOff = r * cols;
          for (let c = 0; c < cols; c++) sum += W.data[rOff + c] * x[c];
          out[r] = sum;
        }
        return out;
      };
      const _nnBatchNorm = (x, weight, bias, mean, variance) => {
        const n = x.length, out = new Float32Array(n);
        for (let i = 0; i < n; i++) out[i] = (x[i] - mean.data[i]) / Math.sqrt(variance.data[i] + 1e-5) * weight.data[i] + bias.data[i];
        return out;
      };
      const _nnRelu = (x) => { const out = new Float32Array(x.length); for (let i = 0; i < x.length; i++) out[i] = x[i] > 0 ? x[i] : 0; return out; };
      const _nnAdd = (a, b) => { const out = new Float32Array(a.length); for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i]; return out; };
      const _nnSoftmax = (x, mask) => {
        const out = new Float32Array(x.length);
        let maxVal = -Infinity;
        for (let i = 0; i < x.length; i++) if (mask[i] > 0 && x[i] > maxVal) maxVal = x[i];
        let sum = 0;
        for (let i = 0; i < x.length; i++) { if (mask[i] > 0) { out[i] = Math.exp(x[i] - maxVal); sum += out[i]; } else out[i] = 0; }
        if (sum > 0) for (let i = 0; i < x.length; i++) out[i] /= sum;
        return out;
      };

      const _nnForward = (state, mask) => {
        const w = _nnModel.weights;
        let h = _nnMatVec(w['input_fc.weight'], w['input_fc.bias'], state);
        h = _nnBatchNorm(h, w['input_bn.weight'], w['input_bn.bias'], w['input_bn.running_mean'], w['input_bn.running_var']);
        h = _nnRelu(h);
        for (let b = 0; b < 4; b++) {
          const pfx = `res_blocks.${b}`;
          let out = _nnMatVec(w[`${pfx}.fc1.weight`], w[`${pfx}.fc1.bias`], h);
          out = _nnBatchNorm(out, w[`${pfx}.bn1.weight`], w[`${pfx}.bn1.bias`], w[`${pfx}.bn1.running_mean`], w[`${pfx}.bn1.running_var`]);
          out = _nnRelu(out);
          out = _nnMatVec(w[`${pfx}.fc2.weight`], w[`${pfx}.fc2.bias`], out);
          out = _nnBatchNorm(out, w[`${pfx}.bn2.weight`], w[`${pfx}.bn2.bias`], w[`${pfx}.bn2.running_mean`], w[`${pfx}.bn2.running_var`]);
          h = _nnRelu(_nnAdd(out, h));
        }
        let p = _nnMatVec(w['policy_fc1.weight'], w['policy_fc1.bias'], h);
        p = _nnBatchNorm(p, w['policy_bn.weight'], w['policy_bn.bias'], w['policy_bn.running_mean'], w['policy_bn.running_var']);
        p = _nnRelu(p);
        p = _nnMatVec(w['policy_fc2.weight'], w['policy_fc2.bias'], p);
        const policy = _nnSoftmax(p, mask);
        let v = _nnMatVec(w['value_fc1.weight'], w['value_fc1.bias'], h);
        v = _nnBatchNorm(v, w['value_bn.weight'], w['value_bn.bias'], w['value_bn.running_mean'], w['value_bn.running_var']);
        v = _nnRelu(v);
        v = _nnMatVec(w['value_fc2.weight'], w['value_fc2.bias'], v);
        return { policy, value: Math.tanh(v[0]) };
      };

      const _nnForwardValue = (state) => {
        const w = _nnModel.weights;
        let h = _nnMatVec(w['input_fc.weight'], w['input_fc.bias'], state);
        h = _nnBatchNorm(h, w['input_bn.weight'], w['input_bn.bias'], w['input_bn.running_mean'], w['input_bn.running_var']);
        h = _nnRelu(h);
        for (let b = 0; b < 4; b++) {
          const pfx = `res_blocks.${b}`;
          let out = _nnMatVec(w[`${pfx}.fc1.weight`], w[`${pfx}.fc1.bias`], h);
          out = _nnBatchNorm(out, w[`${pfx}.bn1.weight`], w[`${pfx}.bn1.bias`], w[`${pfx}.bn1.running_mean`], w[`${pfx}.bn1.running_var`]);
          out = _nnRelu(out);
          out = _nnMatVec(w[`${pfx}.fc2.weight`], w[`${pfx}.fc2.bias`], out);
          out = _nnBatchNorm(out, w[`${pfx}.bn2.weight`], w[`${pfx}.bn2.bias`], w[`${pfx}.bn2.running_mean`], w[`${pfx}.bn2.running_var`]);
          h = _nnRelu(_nnAdd(out, h));
        }
        let v = _nnMatVec(w['value_fc1.weight'], w['value_fc1.bias'], h);
        v = _nnBatchNorm(v, w['value_bn.weight'], w['value_bn.bias'], w['value_bn.running_mean'], w['value_bn.running_var']);
        v = _nnRelu(v);
        v = _nnMatVec(w['value_fc2.weight'], w['value_fc2.bias'], v);
        return Math.tanh(v[0]);
      };

      // Encode game state to 185-dim vector (simplified beliefs using cantHave)
      const _nnEncodeState = (hand, lE, rE, bLen, player, knowledge, matchScores, dobMul) => {
        const state = new Float32Array(NN_STATE_DIM);
        const me = player;
        const partner = (me + 2) % 4, lho = (me + 1) % 4, rho = (me + 3) % 4;
        // [0:28] My hand
        for (const t of hand) state[TILE_INDEX[t.id]] = 1.0;
        // [28:56] Played tiles
        for (const id of knowledge.played) state[28 + TILE_INDEX[id]] = 1.0;
        // [56:63] Left end one-hot
        if (lE >= 0 && lE <= 6) state[56 + lE] = 1.0;
        // [63:70] Right end one-hot
        if (rE >= 0 && rE <= 6) state[63 + rE] = 1.0;
        // [70:91] CantHave: 3 players x 7 numbers
        const others = [partner, lho, rho];
        for (let i = 0; i < 3; i++) {
          for (const n of knowledge.cantHave[others[i]]) {
            if (n >= 0 && n <= 6) state[70 + i * 7 + n] = 1.0;
          }
        }
        // [91:175] Belief probabilities (cantHave-based approximation)
        const handIds = new Set(hand.map(h => h.id));
        for (let t = 0; t < 28; t++) {
          const tile = ALL_TILES[t];
          if (handIds.has(tile.id) || knowledge.played.has(tile.id)) {
            state[91 + t] = state[91 + 28 + t] = state[91 + 56 + t] = 1/3;
            continue;
          }
          // Uniform prior with cantHave elimination
          let b0 = 1, b1 = 1, b2 = 1;
          if (knowledge.cantHave[partner].has(tile.left) || knowledge.cantHave[partner].has(tile.right)) b0 = 0;
          if (knowledge.cantHave[lho].has(tile.left) || knowledge.cantHave[lho].has(tile.right)) b1 = 0;
          if (knowledge.cantHave[rho].has(tile.left) || knowledge.cantHave[rho].has(tile.right)) b2 = 0;
          const total = b0 + b1 + b2;
          if (total > 0) { state[91 + t] = b0/total; state[91+28+t] = b1/total; state[91+56+t] = b2/total; }
          else { state[91 + t] = state[91 + 28 + t] = state[91 + 56 + t] = 1/3; }
        }
        // [175:179] Hand sizes (normalized)
        const playedBy = knowledge.playsBy;
        state[175] = hand.length / 6.0;
        for (let i = 0; i < 3; i++) state[176 + i] = Math.max(0, 6 - playedBy[others[i]].length) / 6.0;
        // [179:181] Match scores
        const ms = matchScores || [0, 0];
        const myTeam = me % 2;
        state[179] = ms[myTeam] / 6.0;
        state[180] = ms[1 - myTeam] / 6.0;
        // [181:182] Multiplier
        state[181] = Math.min(dobMul || 1, 4) / 4.0;
        // [182:183] Board length
        state[182] = bLen / 24.0;
        // [183:184] Game phase
        state[183] = knowledge.played.size / 24.0;
        // [184:185] My team
        state[184] = myTeam;
        return state;
      };

      // Build action mask (57-dim)
      const _nnActionMask = (hand, lE, rE, bLen) => {
        const mask = new Float32Array(NN_NUM_ACTIONS);
        const playable = hand.filter(t => bLen === 0 || t.left === lE || t.right === lE || t.left === rE || t.right === rE);
        if (playable.length === 0) { mask[56] = 1.0; return mask; }
        for (const tile of playable) {
          const idx = TILE_INDEX[tile.id];
          if (bLen === 0) { mask[idx] = 1.0; }
          else if (lE === rE) { mask[idx] = 1.0; }
          else {
            if (tile.left === lE || tile.right === lE) mask[idx] = 1.0;
            if (tile.left === rE || tile.right === rE) mask[28 + idx] = 1.0;
          }
        }
        return mask;
      };

      // === ISMCTS (Information Set Monte Carlo Tree Search) ===
      class ISMCTSNode {
        constructor(move = null, parent = null) {
          this.move = move;
          this.parent = parent;
          this.children = [];
          this.visits = 0;
          this.totalReward = 0;
          this.hBias = 0;
          this.priorP = 0.0;
          this.actionIdx = -1;
        }
        puctScore(parentVisits, cPuct) {
          const Q = this.visits > 0 ? this.totalReward / this.visits : 0.0;
          return Q + cPuct * this.priorP * Math.sqrt(parentVisits) / (1 + this.visits);
        }
        ucb1(C = 1.41, useProgBias = true) {
          if (this.visits === 0) return Infinity;
          return this.totalReward / this.visits + C * Math.sqrt(Math.log(this.parent.visits) / this.visits) + (useProgBias ? this.hBias / (this.visits + 1) : 0);
        }
        selectChild(legalMoves, cfg) {
          const legal = this.children.filter(c =>
            legalMoves.some(m => m.tile.id === c.move.tile.id && m.side === c.move.side)
          );
          if (legal.length === 0) return null;
          const pVisits = Math.max(1, this.visits);
          const doPUCT = cfg && cfg.usePUCT && (!cfg.rootOnlyPUCT || cfg.isRoot);
          let best = legal[0];
          let bestScore = doPUCT ? legal[0].puctScore(pVisits, cfg.cPuct) : legal[0].ucb1(cfg ? cfg.C : 1.41, cfg ? cfg.useProgBias : true);
          for (let i = 1; i < legal.length; i++) {
            const s = doPUCT ? legal[i].puctScore(pVisits, cfg.cPuct) : legal[i].ucb1(cfg ? cfg.C : 1.41, cfg ? cfg.useProgBias : true);
            if (s > bestScore) { best = legal[i]; bestScore = s; }
          }
          return best;
        }
        expand(move) {
          const child = new ISMCTSNode(move, this);
          this.children.push(child);
          return child;
        }
        update(reward) { this.visits++; this.totalReward += reward; }
      }

      const ismctsGetLegalMoves = (hand, lE, rE, bLen) => {
        const playable = hand.filter(t => bLen === 0 || t.left === lE || t.right === lE || t.left === rE || t.right === rE);
        const moves = [];
        for (const tile of playable) {
          if (bLen === 0) {
            moves.push({ tile, side: null });
          } else {
            if (tile.left === lE || tile.right === lE) moves.push({ tile, side: 'left' });
            if ((tile.left === rE || tile.right === rE) && lE !== rE) moves.push({ tile, side: 'right' });
            if (moves.length === 0 || moves[moves.length-1].tile.id !== tile.id) {
              if (tile.left === rE || tile.right === rE) moves.push({ tile, side: 'right' });
            }
          }
        }
        return moves;
      };

      const ismctsApplyMove = (tile, side, lE, rE, bLen) => {
        if (bLen === 0) return { newLE: tile.left, newRE: tile.right };
        if (side === 'left') return { newLE: tile.left === lE ? tile.right : tile.left, newRE: rE };
        return { newLE: lE, newRE: tile.right === rE ? tile.left : tile.right };
      };

      // Heuristic bias for ISMCTS progressive bias
      const _ismctsHeuristic = (move, hand, lE, rE, bLen, knowledge, player) => {
        const { tile, side } = move;
        let h = 0;
        const opp1 = (player + 1) % 4, opp2 = (player + 3) % 4;
        const partner = (player + 2) % 4;
        let newEnd, otherEnd;
        if (bLen === 0) { newEnd = tile.right; otherEnd = tile.left; }
        else if (side === 'left') { newEnd = tile.left === lE ? tile.right : tile.left; otherEnd = rE; }
        else { newEnd = tile.right === rE ? tile.left : tile.right; otherEnd = lE; }
        for (const t of hand) {
          if (t.id !== tile.id && (t.left === newEnd || t.right === newEnd)) h += 0.04;
        }
        if (tile.left === tile.right) h += 0.03;
        if (knowledge.cantHave[opp1].has(newEnd) && knowledge.cantHave[opp1].has(otherEnd)) h += 0.06;
        else if (knowledge.cantHave[opp1].has(newEnd)) h += 0.04;
        if (knowledge.cantHave[opp2].has(newEnd) && knowledge.cantHave[opp2].has(otherEnd)) h += 0.05;
        else if (knowledge.cantHave[opp2].has(newEnd)) h += 0.03;
        if (knowledge.cantHave[partner].has(newEnd) && knowledge.cantHave[partner].has(otherEnd)) h -= 0.03;
        if (bLen > 0) {
          const remNew = knowledge.remainingWithNumber(newEnd);
          if (remNew <= 1 && newEnd !== otherEnd) {
            const tileHasOther = (tile.left === otherEnd || tile.right === otherEnd) ? 1 : 0;
            const remOther = knowledge.remainingWithNumber(otherEnd) - tileHasOther;
            if (remOther > 0) h -= 0.06;
            else h -= 0.02;
          }
        }
        if (hand.length === 1) h += 0.5;
        else if (hand.length === 2) {
          const last = hand.find(t => t.id !== tile.id);
          if (last && (last.left === newEnd || last.right === newEnd || last.left === otherEnd || last.right === otherEnd)) h += 0.12;
          else h -= 0.04;
        } else if (hand.length === 3) {
          let cover = 0;
          for (const t of hand) {
            if (t.id !== tile.id && (t.left === newEnd || t.right === newEnd || t.left === otherEnd || t.right === otherEnd)) cover++;
          }
          if (cover === 2) h += 0.04;
          else if (cover === 0) h -= 0.03;
        }
        const pipMul = bLen <= 4 ? 0.001 : (bLen <= 14 ? 0.003 : 0.005);
        h += (tile.left + tile.right) * pipMul;
        return h;
      };

      const _ismctsBackprop = (node, reward) => {
        while (node) { node.update(reward); node = node.parent; }
      };

      // Score a win for ISMCTS (object-based hands)
      const _ismctsScoreWin = (tile, prevLE, prevRE, boardLen) => {
        const isD = tile.left === tile.right;
        let wasBoth = false;
        if (boardLen > 1 && prevLE >= 0 && prevRE >= 0) {
          if (prevLE === prevRE) {
            wasBoth = (tile.left === prevLE && tile.right === prevLE);
          } else {
            wasBoth = (tile.left === prevLE || tile.right === prevLE) && (tile.left === prevRE || tile.right === prevRE);
          }
        }
        if (isD && wasBoth) return 4;
        if (isD) return 2;
        if (wasBoth) return 3;
        return 1;
      };

      const ismctsEval = (hand, lE, rE, bLen, player, knowledge, matchScores, dobMultiplier) => {
        const MAX_ITERATIONS = 600;
        const TIME_LIMIT = 300;
        const MAX_NODES = 5000;
        const myTeam = player % 2;
        const ms = matchScores || [0, 0];
        const dm = dobMultiplier || 1;

        const playedPerPlayer = knowledge.playsBy.map(a => a.length);
        const handSizes = [0, 0, 0, 0];
        for (let p = 0; p < 4; p++) {
          handSizes[p] = p === player ? hand.length : Math.max(1, 6 - playedPerPlayer[p]);
        }

        // NN-aware config: PUCT at root when model loaded
        const nnActive = USE_NN_LEAF_VALUE && _nnModel;
        const cfg = {
          usePUCT: nnActive,
          rootOnlyPUCT: true,
          cPuct: 1.0,
          C: nnActive ? 0.7 : 1.41,
          useProgBias: !nnActive,
          isRoot: true,
        };

        // Compute root PUCT prior (blended NN policy + uniform)
        let rootPrior = null;
        if (cfg.usePUCT) {
          const mask = _nnActionMask(hand, lE, rE, bLen);
          const enc = _nnEncodeState(hand, lE, rE, bLen, player, knowledge, ms, dm);
          const { policy: pNN } = _nnForward(enc, mask);
          const alpha = 0.2;
          rootPrior = new Float32Array(57);
          let legalCount = 0;
          for (let a = 0; a < 57; a++) if (mask[a] > 0) legalCount++;
          const u = legalCount > 0 ? 1.0 / legalCount : 0;
          let sum = 0;
          for (let a = 0; a < 57; a++) {
            if (mask[a] === 0) { rootPrior[a] = 0; continue; }
            rootPrior[a] = (1 - alpha) * u + alpha * Math.max(pNN[a], 1e-12);
            sum += rootPrior[a];
          }
          if (sum > 0) for (let a = 0; a < 57; a++) rootPrior[a] /= sum;
        }

        const root = new ISMCTSNode();
        let nodeCount = 1;
        const startTime = Date.now();

        for (let iter = 0; iter < MAX_ITERATIONS; iter++) {
          if (Date.now() - startTime > TIME_LIMIT) break;
          if (nodeCount >= MAX_NODES) break;

          // 1. DETERMINIZE: sample a consistent deal
          let deal;
          try { deal = generateConsistentDeal(hand, handSizes, knowledge, player); }
          catch(e) { continue; }

          // 2. SELECT + EXPAND + ROLLOUT
          let node = root;
          let simHands = deal.map(h => [...h]);
          simHands[player] = [...hand];
          let simLE = lE, simRE = rE, simBLen = bLen;
          let simK = {
            cantHave: knowledge.cantHave.map(s => new Set(s)),
            played: new Set(knowledge.played),
            playsBy: knowledge.playsBy.map(a => [...a]),
            _rc: [...Array(7)].map((_, n) => knowledge.remainingWithNumber(n)),
            _sc: [null, null, null, null],
            remainingWithNumber(n) { return this._rc[n]; },
            inferStrength(p) {
              if (this._sc[p]) return this._sc[p];
              const s = [0,0,0,0,0,0,0];
              for (const t of this.playsBy[p]) { s[t.left]++; if (t.left !== t.right) s[t.right]++; }
              this._sc[p] = s; return s;
            },
            recordPlay(p, t) {
              if (!this.played.has(t.id)) {
                this.played.add(t.id);
                this._rc[t.left]--;
                if (t.left !== t.right) this._rc[t.right]--;
              }
              this.playsBy[p].push(t);
              this._sc[p] = null;
            },
            recordPass(p, le, re) { this.cantHave[p].add(le); this.cantHave[p].add(re); }
          };
          let curPlayer = player;

          // Selection phase
          while (node.children.length > 0) {
            const legalMoves = ismctsGetLegalMoves(simHands[curPlayer], simLE, simRE, simBLen);
            if (legalMoves.length === 0) break;

            const selCfg = { ...cfg, isRoot: node === root };
            const child = node.selectChild(legalMoves, selCfg);
            if (!child) break;

            // Progressive widening
            const untriedMoves = legalMoves.filter(m =>
              !node.children.some(c => c.move.tile.id === m.tile.id && c.move.side === m.side)
            );
            if (untriedMoves.length > 0 && node.children.length < Math.ceil(Math.sqrt(node.visits + 1))) {
              const expandMove = untriedMoves[Math.floor(Math.random() * untriedMoves.length)];
              const _prevNode = node;
              node = node.expand(expandMove);
              nodeCount++;
              if (_prevNode === root && curPlayer === player) {
                // Set PUCT prior or heuristic bias
                if (rootPrior) {
                  const tIdx = TILE_INDEX[expandMove.tile.id];
                  node.actionIdx = expandMove.side === 'right' ? 28 + tIdx : tIdx;
                  node.priorP = rootPrior[node.actionIdx];
                }
                if (!cfg.usePUCT) {
                  node.hBias = _ismctsHeuristic(expandMove, hand, lE, rE, bLen, knowledge, player);
                }
              }
              const applied = ismctsApplyMove(expandMove.tile, expandMove.side, simLE, simRE, simBLen);
              simHands[curPlayer] = simHands[curPlayer].filter(t => t.id !== expandMove.tile.id);
              simK.recordPlay(curPlayer, expandMove.tile);
              simLE = applied.newLE; simRE = applied.newRE; simBLen++;
              if (simHands[curPlayer].length === 0) {
                const pts = _ismctsScoreWin(expandMove.tile, lE, rE, simBLen);
                const reward = _rolloutToMEReward(curPlayer % 2, pts, myTeam, ms, dm);
                _ismctsBackprop(node, reward);
                break;
              }
              curPlayer = (curPlayer + 1) % 4;
              break;
            }

            // Follow selected child
            const { tile, side } = child.move;
            const applied = ismctsApplyMove(tile, side, simLE, simRE, simBLen);
            simHands[curPlayer] = simHands[curPlayer].filter(t => t.id !== tile.id);
            simK.recordPlay(curPlayer, tile);
            simLE = applied.newLE; simRE = applied.newRE; simBLen++;
            if (simHands[curPlayer].length === 0) {
              const pts = _ismctsScoreWin(tile, lE, rE, simBLen);
              const reward = _rolloutToMEReward(curPlayer % 2, pts, myTeam, ms, dm);
              _ismctsBackprop(child, reward);
              node = null;
              break;
            }
            curPlayer = (curPlayer + 1) % 4;
            node = child;
          }

          if (node === null) continue;

          // Expansion at leaf
          if (node.children.length === 0) {
            const legalMoves = ismctsGetLegalMoves(simHands[curPlayer], simLE, simRE, simBLen);
            if (legalMoves.length > 0) {
              const expandMove = legalMoves[Math.floor(Math.random() * legalMoves.length)];
              const _leafParent = node;
              node = node.expand(expandMove);
              nodeCount++;
              if (_leafParent === root && curPlayer === player) {
                if (rootPrior) {
                  const tIdx = TILE_INDEX[expandMove.tile.id];
                  node.actionIdx = expandMove.side === 'right' ? 28 + tIdx : tIdx;
                  node.priorP = rootPrior[node.actionIdx];
                }
                if (!cfg.usePUCT) {
                  node.hBias = _ismctsHeuristic(expandMove, hand, lE, rE, bLen, knowledge, player);
                }
              }
              const applied = ismctsApplyMove(expandMove.tile, expandMove.side, simLE, simRE, simBLen);
              simHands[curPlayer] = simHands[curPlayer].filter(t => t.id !== expandMove.tile.id);
              simK.recordPlay(curPlayer, expandMove.tile);
              simLE = applied.newLE; simRE = applied.newRE; simBLen++;
              if (simHands[curPlayer].length === 0) {
                const pts = _ismctsScoreWin(expandMove.tile, lE, rE, simBLen);
                const reward = _rolloutToMEReward(curPlayer % 2, pts, myTeam, ms, dm);
                _ismctsBackprop(node, reward);
                continue;
              }
              curPlayer = (curPlayer + 1) % 4;
            }
          }

          // 3. LEAF EVALUATION: NN value head (if available) or fastAI rollout
          let reward;
          if (nnActive) {
            const nnState = _nnEncodeState(simHands[curPlayer], simLE, simRE, simBLen, curPlayer, simK, ms, dm);
            const value = _nnForwardValue(nnState);
            reward = (curPlayer % 2 === myTeam) ? value : -value;
          } else {
            const result = simulateFromPosition(simHands, simLE, simRE, curPlayer, simK, simBLen);
            const winTeam = result.winnerTeam !== undefined ? result.winnerTeam : result.team;
            reward = (winTeam >= 0) ? _rolloutToMEReward(winTeam, result.points, myTeam, ms, dm) : 0;
          }
          _ismctsBackprop(node, reward);
        }

        // Extract best move from root children
        if (root.children.length === 0) return null;

        const tileResults = new Map();
        for (const child of root.children) {
          if (child.visits === 0) continue;
          const avgReward = child.totalReward / child.visits;
          const tileKey = child.move.tile.id;
          const r = { tile: child.move.tile, side: child.move.side, expectedPoints: avgReward * 4, visits: child.visits };
          if (!tileResults.has(tileKey) || r.expectedPoints > tileResults.get(tileKey).expectedPoints) {
            tileResults.set(tileKey, r);
          }
        }
        const finalResults = [...tileResults.values()];
        finalResults.sort((a, b) => b.expectedPoints - a.expectedPoints || b.visits - a.visits);
        return finalResults.length > 0 ? finalResults[0] : null;
      };

      // === Expert AI router: endgame solver → ISMCTS → MC → smartAI ===
      const expertAI = (hand, lE, rE, bLen, player, knowledge, matchScores, dobMultiplier) => {
        if (bLen === 0) {
          return smartAI(hand, lE, rE, bLen, player, knowledge, matchScores);
        }
        // Count total remaining tiles (not played, not in dormidas)
        const totalRemaining = 24 - (knowledge.played?.size || 0);

        // Endgame: when few tiles remain, exact solver is tractable
        if (totalRemaining <= 16) {
          const egResult = endgameSolve(hand, lE, rE, bLen, player, knowledge, matchScores, dobMultiplier);
          if (egResult) return egResult;
        }

        // Mid-game: ISMCTS tree search for deeper strategic play
        if (bLen >= 6) {
          const ismctsResult = ismctsEval(hand, lE, rE, bLen, player, knowledge, matchScores, dobMultiplier);
          if (ismctsResult) return ismctsResult;
        }

        // Fallback: Monte Carlo rollouts
        const mcResult = monteCarloEval(hand, lE, rE, bLen, player, knowledge);
        if (mcResult) return mcResult;

        // Last resort: heuristic
        return smartAI(hand, lE, rE, bLen, player, knowledge, matchScores);
      };

      // === Easy AI: random valid tile ===
      const randomAI = (hand, lE, rE, bLen) => {
        if (bLen === 0) {
          const t = hand[Math.floor(Math.random() * hand.length)];
          return t ? { tile: t, side: null } : null;
        }
        const playable = [];
        for (const t of hand) {
          if (t.left === lE || t.right === lE) playable.push({ tile: t, side: 'left' });
          if ((t.left === rE || t.right === rE) && lE !== rE) playable.push({ tile: t, side: 'right' });
          if (lE === rE && (t.left === lE || t.right === lE)) { /* already added for left */ }
        }
        return playable.length > 0 ? playable[Math.floor(Math.random() * playable.length)] : null;
      };

      // === Bot AI router — picks AI based on difficulty setting ===
      const botAI = (hand, lE, rE, bLen, player, knowledge, matchScores, dobMultiplier) => {
        const diff = gameState?.config?.aiDifficulty || aiDifficulty;
        if (diff === 'easy') return randomAI(hand, lE, rE, bLen);
        if (diff === 'medium') return smartAI(hand, lE, rE, bLen, player, knowledge, matchScores);
        return expertAI(hand, lE, rE, bLen, player, knowledge, matchScores, dobMultiplier);
      };

      const executeTilePlay = async (tile, slot, side = null) => {
        // Guard against double-click (human players only)
        if (slot === playerSlot) {
          if (playingRef.current) return;
          playingRef.current = true;
          // Safety reset after 2s in case Firebase write hangs
          setTimeout(() => { playingRef.current = false; }, 2000);
        }
        // Determine actual side for fly animation targeting
        let actualSide = side; // 'left', 'right', or null
        if (!actualSide && gameState.board && gameState.board.length > 0) {
          // Auto-placement: figure out which end this tile will go to
          if (tile.left === gameState.leftEnd || tile.right === gameState.leftEnd) {
            actualSide = 'left';
          } else if (tile.left === gameState.rightEnd || tile.right === gameState.rightEnd) {
            actualSide = 'right';
          }
        }
        // null means first tile (board empty) — will center

        // Trigger fly animation before updating state (if enabled)
        if (animations) {
          setFlyingTile({ tile, fromSlot: slot, side: actualSide });
          await new Promise(resolve => setTimeout(resolve, 350));
          setFlyingTile(null);
        }

        const newHands = gameState.hands.map(h => [...h]);
        newHands[slot] = newHands[slot].filter(t => t.id !== tile.id);

        let newBoard = [...(gameState.board || [])];
        let newLeftEnd = gameState.leftEnd;
        let newRightEnd = gameState.rightEnd;

        if (!gameState.board || gameState.board.length === 0) {
          newBoard = [tile];
          newLeftEnd = tile.left;
          newRightEnd = tile.right;
        } else {
          let placedTile = { ...tile };
          if (side === 'left') {
            if (tile.left === gameState.leftEnd) {
              placedTile = { ...tile, left: tile.right, right: tile.left };
            }
            newBoard.unshift(placedTile);
            newLeftEnd = placedTile.left;
          } else if (side === 'right') {
            if (tile.right === gameState.rightEnd) {
              placedTile = { ...tile, left: tile.right, right: tile.left };
            }
            newBoard.push(placedTile);
            newRightEnd = placedTile.right;
          } else {
            // Auto placement
            if (tile.left === gameState.leftEnd) {
              placedTile = { ...tile, left: tile.right, right: tile.left };
              newBoard.unshift(placedTile);
              newLeftEnd = placedTile.left;
            } else if (tile.right === gameState.leftEnd) {
              newBoard.unshift(placedTile);
              newLeftEnd = placedTile.left;
            } else if (tile.left === gameState.rightEnd) {
              newBoard.push(placedTile);
              newRightEnd = placedTile.right;
            } else if (tile.right === gameState.rightEnd) {
              placedTile = { ...tile, left: tile.right, right: tile.left };
              newBoard.push(placedTile);
              newRightEnd = placedTile.right;
            }
          }
        }

        // Check win
        if (newHands[slot].length === 0) {
          await handleWin(slot, tile, newHands, newBoard, newLeftEnd, newRightEnd);
          return;
        }

        const nextPlayer = (slot + 1) % 4;
        const updates = {
          hands: newHands,
          board: newBoard,
          leftEnd: newLeftEnd,
          rightEnd: newRightEnd,
          currentPlayer: nextPlayer,
          passCount: 0,
          lastPass: null,
          message: '',
          moveHistory: [...(gameState.moveHistory || []), {p: slot, t: 'play', tile: {left: tile.left, right: tile.right, id: tile.id}}]
        };

        await db.ref('rooms/' + roomCode).update(updates);
        // Reset guard immediately after successful write (don't wait for 2s timeout)
        if (slot === playerSlot) playingRef.current = false;
      };

      const playTile = async (tile, side = null) => {
        if (gameState.currentPlayer !== playerSlot) return;
        if (!canPlayTile(tile)) return;

        if (canPlayOnBothEnds(tile) && side === null) {
          setChoosingTile(tile);
          return;
        }

        await executeTilePlay(tile, playerSlot, side);
      };

      const handleWin = async (winner, lastTile, hands, board, leftEnd, rightEnd) => {
        const isDouble = lastTile.left === lastTile.right;
        const wasOnBothEnds = couldPlayOnBothEnds(lastTile, gameState.leftEnd, gameState.rightEnd);

        let basePoints, scoreName;
        if (isDouble && wasOnBothEnds) {
          basePoints = 4;
          scoreName = 'cruzada';
        } else if (isDouble) {
          basePoints = 2;
          scoreName = 'com carroca';
        } else if (wasOnBothEnds) {
          basePoints = 3;
          scoreName = 'la e lo';
        } else {
          basePoints = 1;
          scoreName = 'normal';
        }

        const points = basePoints * (gameState.scoreMultiplier || 1);
        const winningTeam = winner % 2;
        const newScores = [...gameState.teamScores];
        newScores[winningTeam] += points;

        const extraMsg = (gameState.scoreMultiplier || 1) > 1 ? ' (' + basePoints + ' x' + (gameState.scoreMultiplier || 1) + ' dobrada)' : '';
        
        const displayName = { 'cruzada': 'CRUZADA!', 'com carroca': 'CARROÇA!', 'la e lo': 'LÁ E LÓ!', 'normal': 'BATEU!' };
        const displayEmoji = { 'cruzada': '💥', 'com carroca': '🎯', 'la e lo': '🔥', 'normal': '✅' };

        const updates = {
          hands: hands,
          board: board,
          leftEnd: leftEnd,
          rightEnd: rightEnd,
          teamScores: newScores,
          scoreMultiplier: 1,
          gameEnded: newScores[winningTeam] >= gameState.matchTarget,
          lastWinningTeam: winningTeam,
          blockedReveal: null,
          roundResult: { scoreName, points, winner, winningTeam, playerName: gameState.players[winner].name },
          message: gameState.players[winner].name + ' bateu ' + scoreName + '! Time ' + (winningTeam + 1) + ' marcou ' + points + ' ponto(s)' + extraMsg + '!',
          currentPlayer: -1
        };

        if (newScores[winningTeam] >= gameState.matchTarget) {
          const losingTeam = 1 - winningTeam;
          const isBuchuda = newScores[losingTeam] === 0;
          updates.roundResult.matchEnd = true;
          updates.roundResult.buchuda = isBuchuda;
          updates.message = isBuchuda
            ? 'BUCHUDA! Time ' + (winningTeam + 1) + ' venceu ' + newScores[winningTeam] + ' a 0!'
            : 'PARTIDA GANHA! Time ' + (winningTeam + 1) + ' venceu com ' + newScores[winningTeam] + ' pontos!';
        }

        await db.ref('rooms/' + roomCode).update(updates);
      };

      const pass = async () => {
        if (gameState.currentPlayer !== playerSlot) return;

        const hasValid = gameState.hands[playerSlot].some(t => canPlayTile(t));
        if (hasValid) {
          setError('Voce tem peca valida!');
          setTimeout(() => setError(''), 2000);
          return;
        }

        const newPassCount = (gameState.passCount || 0) + 1;

        if (newPassCount >= 4) {
          await handleBlocked();
          return;
        }

        const nextPlayer = (playerSlot + 1) % 4;
        await db.ref('rooms/' + roomCode).update({
          passCount: newPassCount,
          currentPlayer: nextPlayer,
          lastPass: playerSlot,
          message: '',
          moveHistory: [...(gameState.moveHistory || []), {p: playerSlot, t: 'pass', lE: gameState.leftEnd, rE: gameState.rightEnd}]
        });
      };

      const botPass = async (botSlot) => {
        const newPassCount = (gameState.passCount || 0) + 1;

        if (newPassCount >= 4) {
          await handleBlocked();
          return;
        }

        const nextPlayer = (botSlot + 1) % 4;
        await db.ref('rooms/' + roomCode).update({
          passCount: newPassCount,
          currentPlayer: nextPlayer,
          lastPass: botSlot,
          message: '',
          moveHistory: [...(gameState.moveHistory || []), {p: botSlot, t: 'pass', lE: gameState.leftEnd, rE: gameState.rightEnd}]
        });
      };

      const handleBlocked = async () => {
        const handValues = gameState.hands.map((hand, idx) => ({
          player: idx,
          points: hand.reduce((sum, t) => sum + t.left + t.right, 0)
        }));

        const minValue = Math.min(...handValues.map(h => h.points));
        const winners = handValues.filter(h => h.points === minValue);
        const handSummary = handValues.map(h => gameState.players[h.player].name + ': ' + h.points).join(' | ');

        // Build blockedReveal data — all players' hands face-up with pip counts
        const blockedReveal = {
          players: handValues.map(h => ({
            slot: h.player,
            name: gameState.players[h.player].name,
            pips: h.points,
            tiles: gameState.hands[h.player]
          })),
          team0Pips: handValues.filter(h => h.player % 2 === 0).reduce((s, h) => s + h.points, 0),
          team1Pips: handValues.filter(h => h.player % 2 === 1).reduce((s, h) => s + h.points, 0)
        };

        if (winners.length > 1 && winners.some(w => w.player % 2 === 0) && winners.some(w => w.player % 2 === 1)) {
          const newMultiplier = (gameState.scoreMultiplier || 1) * 2;
          blockedReveal.isDobrada = true;
          await db.ref('rooms/' + roomCode).update({
            scoreMultiplier: newMultiplier,
            isDobrada: true,
            currentPlayer: -1,
            blockedReveal: blockedReveal,
            message: 'Jogo travado! ' + handSummary + '. Empate! Dobrada! Proximo jogo vale ' + newMultiplier + 'x!'
          });
          return;
        }

        const winner = winners[0].player;
        const points = 1 * (gameState.scoreMultiplier || 1);
        const winningTeam = winner % 2;
        const newScores = [...gameState.teamScores];
        newScores[winningTeam] += points;

        blockedReveal.winningTeam = winningTeam;
        blockedReveal.winnerSlot = winner;

        let blockedMsg = 'Jogo travado! ' + handSummary + '. ' + gameState.players[winner].name + ' ganhou! Time ' + (winningTeam + 1) + ' marcou ' + points + '!';
        if (newScores[winningTeam] >= gameState.matchTarget) {
          const losingTeam = 1 - winningTeam;
          const isBuchuda = newScores[losingTeam] === 0;
          blockedMsg = isBuchuda
            ? 'BUCHUDA! ' + handSummary + '. Time ' + (winningTeam + 1) + ' venceu ' + newScores[winningTeam] + ' a 0!'
            : 'PARTIDA GANHA! ' + handSummary + '. Time ' + (winningTeam + 1) + ' venceu com ' + newScores[winningTeam] + ' pontos!';
        }

        await db.ref('rooms/' + roomCode).update({
          teamScores: newScores,
          scoreMultiplier: 1,
          currentPlayer: -1,
          lastWinningTeam: winningTeam,
          gameEnded: newScores[winningTeam] >= gameState.matchTarget,
          blockedReveal: blockedReveal,
          message: blockedMsg
        });
      };

      const newRound = async () => {
        const deck = shuffleDeck(createDeck());
        const hands = [[], [], [], []];
        for (let i = 0; i < 24; i++) {
          hands[i % 4].push(deck[i]);
        }
        const dormidas = deck.slice(24, 28);

        // After dobrada (tie): highest double holder starts, auto-play it
        if (gameState.isDobrada) {
          let startPlayer = 0, highestDouble = -1, highestDoubleTile = null;
          for (let p = 0; p < 4; p++) {
            for (let tile of hands[p]) {
              if (tile.left === tile.right && tile.left > highestDouble) {
                highestDouble = tile.left;
                startPlayer = p;
                highestDoubleTile = tile;
              }
            }
          }
          if (!highestDoubleTile) {
            // Safety redeal (virtually impossible)
            await newRound();
            return;
          }
          hands[startPlayer] = hands[startPlayer].filter(t => t.id !== highestDoubleTile.id);
          const nextPlayer = (startPlayer + 1) % 4;
          await db.ref('rooms/' + roomCode).update({
            hands, dormidas, board: [highestDoubleTile],
            leftEnd: highestDoubleTile.left, rightEnd: highestDoubleTile.right,
            currentPlayer: nextPlayer, passCount: 0,
            isDobrada: false,
            waitingForStarterChoice: false,
            blockedReveal: null,
            roundResult: null,
            moveHistory: [{p: startPlayer, t: 'play', tile: highestDoubleTile}],
            message: gameState.players[startPlayer].name + ' jogou a carroca ' + highestDouble + '-' + highestDouble + '! Vez de ' + gameState.players[nextPlayer].name + '!'
          });
          return;
        }

        // Normal win: winning team chooses starter
        const winTeam = gameState.lastWinningTeam;
        const winTeamSlots = winTeam === 0 ? [0, 2] : [1, 3];
        const winTeamHasHuman = winTeamSlots.some(s => gameState.players[s]?.isHuman);

        if (winTeamHasHuman) {
          await db.ref('rooms/' + roomCode).update({
            hands, dormidas, board: [], leftEnd: null, rightEnd: null,
            currentPlayer: -1, passCount: 0,
            isDobrada: false,
            waitingForStarterChoice: true,
            starterChoiceDeadline: Date.now() + 30000,
            blockedReveal: null,
            roundResult: null,
            moveHistory: [],
            message: 'Time ' + (winTeam + 1) + ' venceu! Veja suas pecas e escolha quem comeca (30s)!'
          });
        } else {
          // Bot winning team — pick best opener from winning team only
          let bestSlot = winTeamSlots[0], bestScore = -1;
          for (const s of winTeamSlots) {
            const sc = [0,0,0,0,0,0,0];
            for (const t of hands[s]) { sc[t.left]++; if (t.left !== t.right) sc[t.right]++; }
            for (const t of hands[s]) {
              let tScore = (t.left === t.right ? 100 : 0) + Math.max(sc[t.left], sc[t.right]) * 10 + t.left + t.right;
              if (tScore > bestScore) { bestScore = tScore; bestSlot = s; }
            }
          }
          await db.ref('rooms/' + roomCode).update({
            hands, dormidas, board: [], leftEnd: null, rightEnd: null,
            currentPlayer: bestSlot, passCount: 0,
            isDobrada: false,
            waitingForStarterChoice: false,
            blockedReveal: null,
            roundResult: null,
            moveHistory: [],
            message: gameState.players[bestSlot].name + ' comeca!'
          });
        }
      };

      // Pass badge on avatar
      useEffect(() => {
        if (gameState?.lastPass == null) { setPassedSlot(null); return; }
        setPassedSlot(gameState.lastPass);
        const t = setTimeout(() => setPassedSlot(null), 2500);
        return () => clearTimeout(t);
      }, [gameState?.lastPass, gameState?.currentPlayer]);

      // Round result announcement overlay
      useEffect(() => {
        if (!gameState?.roundResult) { setRoundAnnouncement(null); return; }
        const r = gameState.roundResult;
        const labels = { 'cruzada': 'CRUZADA!', 'com carroca': 'CARROÇA!', 'la e lo': 'LÁ E LÓ!', 'normal': 'BATEU!' };
        const emojis = { 'cruzada': '💥', 'com carroca': '🎯', 'la e lo': '🔥', 'normal': '✅' };
        setRoundAnnouncement({
          label: labels[r.scoreName] || 'BATEU!',
          emoji: emojis[r.scoreName] || '✅',
          points: r.points,
          playerName: r.playerName,
          scoreName: r.scoreName
        });
        const t = setTimeout(() => setRoundAnnouncement(null), 3500);
        return () => clearTimeout(t);
      }, [gameState?.roundResult?.scoreName, gameState?.currentPlayer]);

      // === Player Statistics Tracking ===
      useEffect(() => {
        if (!gameState || !gameState.players || !gameState.gameStarted) return;

        const initPlayer = (name, team) => {
          if (!playerStatsRef.current[name]) {
            playerStatsRef.current[name] = {
              name, team,
              matchesWon: 0, matchesLost: 0,
              roundsWon: 0, roundsLost: 0,
              winTypes: { normal: 0, cruzada: 0, 'com carroca': 0, 'la e lo': 0, blocked: 0 },
              buchudaGiven: 0, buchudaReceived: 0,
              totalPoints: 0
            };
          }
          playerStatsRef.current[name].team = team;
        };

        // Init all 4 players
        for (let i = 0; i < 4; i++) {
          if (gameState.players[i]) {
            initPlayer(gameState.players[i].name, i % 2);
          }
        }

        // Track round results (normal win via roundResult)
        const rr = gameState.roundResult;
        if (rr && rr.winningTeam != null) {
          const roundKey = rr.playerName + '_' + rr.scoreName + '_' + rr.points + '_' + (gameState.teamScores?.[0]||0) + '_' + (gameState.teamScores?.[1]||0);
          if (lastTrackedRoundRef.current !== roundKey) {
            lastTrackedRoundRef.current = roundKey;
            const wt = rr.winningTeam;
            const lt = 1 - wt;
            // Round won/lost for each player
            for (let i = 0; i < 4; i++) {
              const pName = gameState.players[i]?.name;
              if (!pName || !playerStatsRef.current[pName]) continue;
              if (i % 2 === wt) {
                playerStatsRef.current[pName].roundsWon++;
              } else {
                playerStatsRef.current[pName].roundsLost++;
              }
            }
            // Win type for winning team players
            const sn = rr.scoreName || 'normal';
            for (let i = 0; i < 4; i++) {
              const pName = gameState.players[i]?.name;
              if (!pName || !playerStatsRef.current[pName]) continue;
              if (i % 2 === wt) {
                playerStatsRef.current[pName].winTypes[sn] = (playerStatsRef.current[pName].winTypes[sn] || 0) + 1;
                playerStatsRef.current[pName].totalPoints += (rr.points || 0);
              }
            }
          }
        }

        // Track blocked game results (no roundResult, but blockedReveal with winningTeam)
        const br = gameState.blockedReveal;
        if (br && br.winningTeam != null && !br.isDobrada) {
          const blockedKey = 'blocked_' + br.winnerSlot + '_' + (gameState.teamScores?.[0]||0) + '_' + (gameState.teamScores?.[1]||0);
          if (lastTrackedRoundRef.current !== blockedKey) {
            lastTrackedRoundRef.current = blockedKey;
            const wt = br.winningTeam;
            for (let i = 0; i < 4; i++) {
              const pName = gameState.players[i]?.name;
              if (!pName || !playerStatsRef.current[pName]) continue;
              if (i % 2 === wt) {
                playerStatsRef.current[pName].roundsWon++;
                playerStatsRef.current[pName].winTypes.blocked = (playerStatsRef.current[pName].winTypes.blocked || 0) + 1;
                playerStatsRef.current[pName].totalPoints += 1;
              } else {
                playerStatsRef.current[pName].roundsLost++;
              }
            }
          }
        }

        // Track match end
        if (gameState.gameEnded && gameState.teamScores) {
          const s0 = gameState.teamScores[0] || 0;
          const s1 = gameState.teamScores[1] || 0;
          const matchKey = 'match_' + s0 + '_' + s1 + '_' + Date.now().toString(36).slice(0, 6);
          // Use a stable key based on scores — but since scores reset on new match, we track via gameEnded flag
          const stableMatchKey = 'match_' + s0 + '_' + s1;
          if (lastTrackedMatchRef.current !== stableMatchKey) {
            lastTrackedMatchRef.current = stableMatchKey;
            const mt = gameState.matchTarget || 6;
            const winTeam = s0 >= mt ? 0 : 1;
            const loseTeam = 1 - winTeam;
            const isBuchuda = (winTeam === 0 ? s1 : s0) === 0;

            for (let i = 0; i < 4; i++) {
              const pName = gameState.players[i]?.name;
              if (!pName || !playerStatsRef.current[pName]) continue;
              if (i % 2 === winTeam) {
                playerStatsRef.current[pName].matchesWon++;
                if (isBuchuda) playerStatsRef.current[pName].buchudaGiven++;
              } else {
                playerStatsRef.current[pName].matchesLost++;
                if (isBuchuda) playerStatsRef.current[pName].buchudaReceived++;
              }
            }
          }
        }

        // Reset match tracking when a new game starts (scores back to 0)
        if (!gameState.gameEnded && gameState.teamScores && gameState.teamScores[0] === 0 && gameState.teamScores[1] === 0) {
          lastTrackedMatchRef.current = null;
        }
      }, [gameState?.roundResult, gameState?.blockedReveal, gameState?.gameEnded, gameState?.teamScores]);

      // Animate score dial needle sweep — slow + pulse highlight
      useEffect(() => {
        const s0 = gameState?.teamScores?.[0] || 0;
        const s1 = gameState?.teamScores?.[1] || 0;
        if (s0 !== prevScore0Ref.current || s1 !== prevScore1Ref.current) {
          const from0 = prevScore0Ref.current, from1 = prevScore1Ref.current;
          const changed0 = s0 !== from0, changed1 = s1 !== from1;
          prevScore0Ref.current = s0;
          prevScore1Ref.current = s1;
          // Pulse the dial that changed
          setDialPulse(changed0 ? 'team0' : 'team1');
          const dur = 2000, start = performance.now();
          const tick = (now) => {
            const t = Math.min((now - start) / dur, 1);
            const ease = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
            setAnimScore0(from0 + (s0 - from0) * ease);
            setAnimScore1(from1 + (s1 - from1) * ease);
            if (t < 1) requestAnimationFrame(tick);
            else setDialPulse(null);
          };
          requestAnimationFrame(tick);
        }
      }, [gameState?.teamScores?.[0], gameState?.teamScores?.[1]]);

      // Auto-start next round — 20s countdown, show button over board after 10s
      useEffect(() => {
        const shouldCount = gameState && gameState.gameStarted && !gameState.gameEnded
          && gameState.currentPlayer === -1 && !gameState.waitingForStarterChoice && playerSlot === 0;
        if (!shouldCount) { setRoundCountdown(null); setShowNextBtn(false); return; }
        const delay = 20;
        setRoundCountdown(delay);
        setShowNextBtn(false);
        const btnTimer = setTimeout(() => setShowNextBtn(true), 10000);
        const iv = setInterval(() => {
          setRoundCountdown(prev => {
            if (prev <= 1) { clearInterval(iv); newRound(); return null; }
            return prev - 1;
          });
        }, 1000);
        return () => { clearInterval(iv); clearTimeout(btnTimer); };
      }, [gameState?.currentPlayer, gameState?.gameEnded, gameState?.waitingForStarterChoice]);

      // Bot AI - runs when it's a bot's turn
      useEffect(() => {
        if (!gameState || !gameState.gameStarted || gameState.gameEnded) return;
        if (gameState.currentPlayer === -1) return;
        if (gameState.waitingForStarterChoice) return;
        
        const currentSlot = gameState.currentPlayer;
        const isBot = gameState.players[currentSlot] && !gameState.players[currentSlot].isHuman;
        
        if (!isBot) return;

        // Only player 0 (host) handles bot moves to avoid duplicates
        if (playerSlot !== 0) return;

        // Check if bot can play before deciding delay
        const botHand = gameState.hands[currentSlot];
        const canPlay = botHand.some(t =>
          t.left === gameState.leftEnd || t.right === gameState.leftEnd ||
          t.left === gameState.rightEnd || t.right === gameState.rightEnd
        );
        // If bot can't play, pass immediately (no artificial delay)
        const delay = !canPlay ? 300
          : (() => { const spd = gameState?.config?.botSpeed || botSpeed; return spd === 'instant' ? 100 : spd === 'slow' ? 9000 + Math.random() * 2000 : spd === 'fast' ? 2000 + Math.random() * 2000 : 4000 + Math.random() * 2000; })();

        const timeout = setTimeout(async () => {
          const knowledge = buildKnowledge(gameState.moveHistory);
          const bLen = gameState.board?.length || 0;

          const matchScores = gameState.teamScores || [0, 0];
          const dobMultiplier = gameState.scoreMultiplier || 1;
          const result = botAI(botHand, gameState.leftEnd, gameState.rightEnd, bLen, currentSlot, knowledge, matchScores, dobMultiplier);

          if (result) {
            await executeTilePlay(result.tile, currentSlot, result.side);
          } else {
            await botPass(currentSlot);
          }
        }, delay);

        return () => clearTimeout(timeout);
      }, [gameState?.currentPlayer, gameState?.gameStarted]);

      // Auto-pass: if it's the human's turn and no valid moves, auto-pass after 1.5s
      useEffect(() => {
        if (!gameState || !gameState.gameStarted || gameState.gameEnded) return;
        if (gameState.currentPlayer !== playerSlot) return;
        if (!gameState.players?.[playerSlot]?.isHuman) return;
        if (!gameState.board || gameState.board.length === 0) return;
        const hasValid = gameState.hands[playerSlot].some(t => canPlayTile(t));
        if (hasValid) return;
        const timeout = setTimeout(() => {
          pass();
        }, 1500);
        return () => clearTimeout(timeout);
      }, [gameState?.currentPlayer, gameState?.gameStarted]);

      // Move timer: 30 second countdown for human player, auto-play random valid tile on timeout
      useEffect(() => {
        playingRef.current = false;  // reset double-click guard on turn change
        if (!gameState || !gameState.gameStarted || gameState.gameEnded) return;
        if (gameState.currentPlayer !== playerSlot) { setMoveTimer(null); return; }
        if (!gameState.players?.[playerSlot]?.isHuman) { setMoveTimer(null); return; }
        const hasValid = gameState.hands[playerSlot].some(t => canPlayTile(t));
        if (!hasValid) { setMoveTimer(null); return; } // auto-pass handles this
        playSound('turn');
        setMoveTimer(30);
        const iv = setInterval(() => {
          setMoveTimer(prev => {
            if (prev <= 1) {
              clearInterval(iv);
              // Auto-play: pick first valid tile
              const hand = gameState.hands[playerSlot];
              const validTile = hand.find(t => canPlayTile(t));
              if (validTile) {
                if (canPlayOnBothEnds(validTile)) {
                  playTile(validTile, 'left');
                } else {
                  playTile(validTile);
                }
              }
              return null;
            }
            return prev - 1;
          });
        }, 1000);
        return () => { clearInterval(iv); };
      }, [gameState?.currentPlayer, gameState?.gameStarted]);

      // Auto-start for solo mode (1 human)
      useEffect(() => {
        if (!gameState || gameState.gameStarted) return;
        const cfg = gameState.config;
        if (!cfg || cfg.humanCount !== 1) return;
        if (playerSlot !== 0) return;
        // All slots are filled (slot 0 is creator, rest are bots), start immediately
        const allReady = cfg.humanSlots.every(s => gameState.players?.[s]);
        if (allReady) {
          startGame();
        }
      }, [gameState?.config, gameState?.gameStarted]);

      // Starter choice countdown display
      useEffect(() => {
        if (!gameState?.waitingForStarterChoice || !gameState?.starterChoiceDeadline) {
          setStarterCountdown(null);
          return;
        }
        const updateCountdown = () => {
          const remaining = Math.max(0, Math.ceil((gameState.starterChoiceDeadline - Date.now()) / 1000));
          setStarterCountdown(remaining);
        };
        updateCountdown();
        const interval = setInterval(updateCountdown, 1000);
        return () => clearInterval(interval);
      }, [gameState?.waitingForStarterChoice, gameState?.starterChoiceDeadline]);

      // Resolve starter votes: when both humans on winning team have voted
      useEffect(() => {
        if (!gameState?.waitingForStarterChoice) return;
        if (playerSlot !== 0) return; // only host resolves
        const votes = gameState.starterVotes;
        if (!votes) return;
        const winTeam = gameState.lastWinningTeam;
        const winSlots = winTeam === 0 ? [0, 2] : [1, 3];
        const humanSlots = winSlots.filter(s => gameState.players?.[s]?.isHuman);
        if (humanSlots.length < 2) return; // solo human — direct pick, no voting needed
        const v0 = votes[humanSlots[0]], v1 = votes[humanSlots[1]];
        if (v0 === undefined || v1 === undefined) return; // waiting for both
        if (v0 === v1) {
          // Both agree on the same player
          startGameWithStarter(v0);
        } else {
          // Conflict — tiebreak: fewer pips starts
          const pips0 = handPipCount(gameState.hands?.[humanSlots[0]] || []);
          const pips1 = handPipCount(gameState.hands?.[humanSlots[1]] || []);
          startGameWithStarter(pips0 <= pips1 ? humanSlots[0] : humanSlots[1]);
        }
      }, [gameState?.starterVotes, gameState?.waitingForStarterChoice]);

      // Auto-pick starter after timer expires (host only) — pip tiebreak, no randomness
      useEffect(() => {
        if (!gameState?.waitingForStarterChoice || !gameState?.starterChoiceDeadline) return;
        if (playerSlot !== 0) return;
        const pickByPips = () => {
          const winTeam = gameState.lastWinningTeam;
          const winSlots = winTeam === 0 ? [0, 2] : [1, 3];
          const pips0 = handPipCount(gameState.hands?.[winSlots[0]] || []);
          const pips1 = handPipCount(gameState.hands?.[winSlots[1]] || []);
          startGameWithStarter(pips0 <= pips1 ? winSlots[0] : winSlots[1]);
        };
        const timeLeft = gameState.starterChoiceDeadline - Date.now();
        if (timeLeft <= 0) { pickByPips(); return; }
        const timeout = setTimeout(pickByPips, timeLeft);
        return () => clearTimeout(timeout);
      }, [gameState?.waitingForStarterChoice, gameState?.starterChoiceDeadline]);

      /* Lock board dimensions: measure ONCE when entering game, then never change */
      useEffect(() => {
        if (screen !== 'game') return;
        if (!boardRef.current) return;
        let cancelled = false;
        let tries = 0;
        const measureAndLock = () => {
          if (cancelled) return;
          const el = boardRef.current;
          if (!el) return;
          const w = Math.floor(el.clientWidth);
          const h = Math.floor(el.clientHeight);
          if ((w < 50 || h < 50) && tries < 30) { tries++; requestAnimationFrame(measureAndLock); return; }
          setBoardBox(prev => (prev.w === w && prev.h === h) ? prev : { w, h });
        };
        requestAnimationFrame(() => requestAnimationFrame(measureAndLock));
        return () => { cancelled = true; };
      }, [screen]);

      // Domino dots pattern
      /* Hand tiles same size as board tiles (vertical orientation: vw × hw) */
      const hDims = { w: bDims.vw, h: bDims.hw, dot: bDims.vw <= 22 ? 3 : bDims.vw <= 26 ? 4 : 5, cont: bDims.vw <= 22 ? 15 : bDims.vw <= 26 ? 18 : 22 };

      const BoardTile = ({ tile, orientation, flipped, extraStyle, hw: hwOverride, vw: vwOverride }) => {
        const isVertical = orientation === 'vertical';
        const val1 = flipped ? tile.right : tile.left;
        const val2 = flipped ? tile.left : tile.right;
        const dividerStyle = isVertical
          ? { borderBottom: '1px solid #b8a888', width: '100%' }
          : { borderRight: '1px solid #b8a888', height: '100%' };
        const hw = hwOverride || bDims.hw, vw = vwOverride || bDims.vw;
        const bDotPx = vw <= 16 ? 2 : vw <= 22 ? 3 : vw <= 26 ? 4 : 5;
        const bContPx = vw <= 16 ? 12 : vw <= 22 ? 15 : vw <= 26 ? 18 : 22;
        return (
          <div
            className="board-tile inline-flex"
            style={Object.assign({}, isVertical
              ? { width: vw, height: hw, flexDirection: 'column' }
              : { width: hw, height: vw, flexDirection: 'row' }, extraStyle || {})}
          >
            <div className="flex-1 flex items-center justify-center">
              <DominoDots value={val1} dotPxProp={bDotPx} containerPxProp={bContPx} />
            </div>
            <div style={dividerStyle}></div>
            <div className="flex-1 flex items-center justify-center">
              <DominoDots value={val2} dotPxProp={bDotPx} containerPxProp={bContPx} />
            </div>
          </div>
        );
      };

      // Confetti helper
      const launchConfetti = () => {
        const colors = ['#f59e0b','#ef4444','#22c55e','#3b82f6','#a855f7','#ec4899'];
        for (let i = 0; i < 60; i++) {
          const el = document.createElement('div');
          el.className = 'confetti-piece';
          el.style.left = Math.random() * 100 + 'vw';
          el.style.background = colors[Math.floor(Math.random() * colors.length)];
          el.style.animationDuration = (2 + Math.random() * 2) + 's';
          el.style.animationDelay = Math.random() * 0.5 + 's';
          el.style.width = (6 + Math.random() * 8) + 'px';
          el.style.height = (6 + Math.random() * 8) + 'px';
          document.body.appendChild(el);
          setTimeout(() => el.remove(), 4500);
        }
      };

      // Menu Screen
      if (screen === 'menu') {
        return (
          <div className="min-h-screen felt-bg p-4 flex items-center justify-center">
            <div className="glass-card p-8 w-full max-w-sm animate-fade-in">
              {/* Logo/title area */}
              <div className="text-center mb-6">
                <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl mb-3" style={{ background: 'linear-gradient(135deg, #166534, #15803d)' }}>
                  <span className="text-3xl">🁣</span>
                </div>
                <h1 className="text-2xl font-extrabold text-gray-800">Domino Pernambucano</h1>
                <p className="text-gray-500 text-sm mt-1">Jogo de domino 2v2</p>
              </div>

              {/* Profile selection */}
              <div className="mb-4">
                <label className="block text-sm font-semibold text-gray-600 mb-2">Escolha seu perfil</label>
                <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', justifyContent: 'center' }}>
                  {HUMAN_PROFILES.map(profile => (
                    <button
                      key={profile.key}
                      onClick={() => { setSelectedProfile(profile); setPlayerName(profile.name); }}
                      style={{
                        display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6, padding: 12, borderRadius: 16,
                        transition: 'all 0.2s',
                        background: selectedProfile?.key === profile.key ? 'linear-gradient(135deg, #dcfce7, #bbf7d0)' : '#f9fafb',
                        border: selectedProfile?.key === profile.key ? '3px solid #22c55e' : '2px solid #e5e7eb',
                        boxShadow: selectedProfile?.key === profile.key ? '0 4px 12px rgba(34,197,94,0.25)' : 'none',
                        transform: selectedProfile?.key === profile.key ? 'scale(1.05)' : 'scale(1)',
                        cursor: 'pointer', minWidth: 80,
                      }}
                    >
                      <Avatar profile={profile} size={56} />
                      <span style={{ fontSize: 13, fontWeight: 700, color: selectedProfile?.key === profile.key ? '#16a34a' : '#6b7280' }}>
                        {profile.name}
                      </span>
                    </button>
                  ))}
                </div>
              </div>

              <div className="mb-5">
                <label className="block text-sm font-semibold text-gray-600 mb-2">Jogadores humanos</label>
                <div className="flex gap-2">
                  {[1, 2, 3, 4].map(n => (
                    <button
                      key={n}
                      onClick={() => setHumanCount(n)}
                      className="flex-1 py-2.5 rounded-xl font-bold text-sm transition-all duration-200"
                      style={humanCount === n
                        ? { background: 'linear-gradient(135deg, #22c55e, #16a34a)', color: 'white', boxShadow: '0 4px 12px rgba(22,163,74,0.3)' }
                        : { background: '#f3f4f6', color: '#4b5563' }}
                    >
                      {n}
                    </button>
                  ))}
                </div>
                <p className="text-xs text-gray-400 mt-2 text-center">
                  {humanCount === 1 && '1 humano + 3 bots (solo)'}
                  {humanCount === 2 && '2 humanos (parceiros) + 2 bots'}
                  {humanCount === 3 && '3 humanos + 1 bot'}
                  {humanCount === 4 && '4 humanos (sem bots)'}
                </p>
              </div>

              {humanCount < 4 && (
              <div className="mb-5">
                <label className="block text-sm font-semibold text-gray-600 mb-2">Forca da IA (bots)</label>
                <div className="flex gap-2">
                  {[{k:'easy',label:'Facil',emoji:'😊'},{k:'medium',label:'Medio',emoji:'🤔'},{k:'hard',label:'Dificil',emoji:'🔥'}].map(d => (
                    <button
                      key={d.k}
                      onClick={() => setAiDiff(d.k)}
                      className="flex-1 py-2.5 rounded-xl font-bold text-sm transition-all duration-200"
                      style={aiDifficulty === d.k
                        ? { background: d.k === 'easy' ? 'linear-gradient(135deg, #22c55e, #16a34a)' : d.k === 'medium' ? 'linear-gradient(135deg, #f59e0b, #d97706)' : 'linear-gradient(135deg, #ef4444, #b91c1c)', color: 'white', boxShadow: '0 4px 12px rgba(0,0,0,0.2)' }
                        : { background: '#f3f4f6', color: '#4b5563' }}
                    >
                      {d.emoji} {d.label}
                    </button>
                  ))}
                </div>
                <p className="text-xs text-gray-400 mt-2 text-center">
                  {aiDifficulty === 'easy' && 'Bots jogam pecas aleatorias'}
                  {aiDifficulty === 'medium' && 'Bots usam estrategia basica'}
                  {aiDifficulty === 'hard' && 'Bots usam IA completa com busca'}
                </p>
              </div>
              )}

              {humanCount < 4 && (
              <div className="mb-5">
                <label className="block text-sm font-semibold text-gray-600 mb-2">Velocidade dos Bots</label>
                <div className="flex gap-2">
                  {[{k:'slow',label:'Lento',emoji:'🐢'},{k:'medium',label:'Medio',emoji:'⚡'},{k:'fast',label:'Rapido',emoji:'🚀'},{k:'instant',label:'Teste',emoji:'⏩'}].map(d => (
                    <button
                      key={d.k}
                      onClick={() => setBotSpd(d.k)}
                      className="flex-1 py-2.5 rounded-xl font-bold text-sm transition-all duration-200"
                      style={botSpeed === d.k
                        ? { background: d.k === 'slow' ? 'linear-gradient(135deg, #22c55e, #16a34a)' : d.k === 'medium' ? 'linear-gradient(135deg, #f59e0b, #d97706)' : d.k === 'instant' ? 'linear-gradient(135deg, #8b5cf6, #6d28d9)' : 'linear-gradient(135deg, #ef4444, #b91c1c)', color: 'white', boxShadow: '0 4px 12px rgba(0,0,0,0.2)' }
                        : { background: '#f3f4f6', color: '#4b5563' }}
                    >
                      {d.emoji} {d.label}
                    </button>
                  ))}
                </div>
                <p className="text-xs text-gray-400 mt-2 text-center">
                  {botSpeed === 'slow' && 'Bots pensam ~10 segundos'}
                  {botSpeed === 'medium' && 'Bots pensam ~5 segundos'}
                  {botSpeed === 'fast' && 'Bots pensam ~3 segundos'}
                  {botSpeed === 'instant' && 'Bots jogam instantaneamente (teste)'}
                </p>
              </div>
              )}

              <div className="mb-5 flex items-center justify-between">
                <label className="text-sm font-semibold text-gray-600">Animações</label>
                <button
                  onClick={() => setAnim(!animations)}
                  className="relative w-12 h-7 rounded-full transition-all duration-200"
                  style={{ background: animations ? 'linear-gradient(135deg, #22c55e, #16a34a)' : '#d1d5db' }}
                >
                  <div className="absolute top-0.5 w-6 h-6 bg-white rounded-full shadow transition-all duration-200"
                    style={{ left: animations ? 22 : 2 }} />
                </button>
              </div>

              <button onClick={createRoom} className="btn-primary w-full mb-3 text-lg">
                Criar Sala
              </button>

              <div className="flex items-center gap-3 my-4">
                <div className="flex-1 h-px bg-gray-200"></div>
                <span className="text-gray-400 text-sm font-semibold">OU</span>
                <div className="flex-1 h-px bg-gray-200"></div>
              </div>

              <input
                type="text"
                placeholder="Codigo da sala"
                value={inputCode}
                onChange={(e) => setInputCode(e.target.value.toUpperCase())}
                className="input-field mb-3 text-center tracking-widest font-bold"
                maxLength={5}
              />

              <button onClick={joinRoom} className="btn-secondary w-full text-lg">
                Entrar na Sala
              </button>

              {error && (
                <div className="mt-4 p-3 rounded-xl bg-red-50 border border-red-200 text-red-600 text-center text-sm font-semibold animate-slide-down">
                  {error}
                </div>
              )}
            </div>
          </div>
        );
      }

      // Lobby Screen
      if (screen === 'lobby') {
        const players = gameState?.players || {};
        const cfg = gameState?.config || { humanCount: 2, humanSlots: [0, 2] };
        const allHumansJoined = cfg.humanSlots.every(s => players[s] !== null && players[s] !== undefined);
        const slotLabels = { 0: 'Jogador 1', 1: 'Jogador 2', 2: 'Jogador 3', 3: 'Jogador 4' };
        const teamSlots = { 0: [0, 2], 1: [1, 3] };
        const teamColors = { 0: { bg: '#dbeafe', border: '#93c5fd', text: '#1e40af', label: '#3b82f6' }, 1: { bg: '#fde8e8', border: '#fca5a5', text: '#991b1b', label: '#ef4444' } };

        return (
          <div className="min-h-screen felt-bg p-4 flex items-start justify-center pt-8">
            <div className="max-w-sm w-full animate-fade-in">
              {/* Room code card */}
              <div className="glass-card p-6 mb-4 text-center">
                <p className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-1">Codigo da Sala</p>
                <div className="text-4xl font-extrabold tracking-[0.3em] mb-2" style={{ color: '#16a34a' }}>{roomCode}</div>
                {cfg.humanCount > 1 && (
                  <p className="text-gray-500 text-sm">Envie este codigo para os outros jogadores!</p>
                )}
                <div className="mt-2 inline-block px-3 py-1 rounded-full text-xs font-bold" style={{ background: '#f0fdf4', color: '#166534' }}>
                  {cfg.humanCount} humano(s) + {4 - cfg.humanCount} bot(s)
                </div>
              </div>

              {/* Teams card */}
              <div className="glass-card p-5 mb-4">
                {[0, 1].map(team => (
                  <div key={team} className={team === 1 ? 'mt-5 pt-5 border-t border-gray-200' : ''}>
                    <div className="flex items-center gap-2 mb-3">
                      <div className="w-3 h-3 rounded-full" style={{ background: teamColors[team].label }}></div>
                      <h3 className="font-bold text-gray-800">Time {team + 1}</h3>
                    </div>
                    <div className="space-y-2">
                      {teamSlots[team].map(slot => {
                        const player = players[slot];
                        const tc = teamColors[team];
                        const prof = profileFromPlayer(player);
                        return (
                          <div key={slot} className="flex items-center gap-3 p-3 rounded-xl transition-all duration-300" style={{ background: tc.bg, border: '1px solid ' + tc.border }}>
                            {player ? (
                              <>
                                <Avatar profile={prof} size={36} />
                                <div className="flex-1 min-w-0">
                                  <div className="font-bold text-sm truncate" style={{ color: player.isHuman ? '#16a34a' : tc.text }}>
                                    {player.name}
                                  </div>
                                  <div className="text-[10px] font-semibold" style={{ color: player.isHuman ? '#22c55e' : tc.text + '99' }}>
                                    {player.isHuman ? (slot === playerSlot ? 'Voce' : 'Humano') : 'Bot'}
                                  </div>
                                </div>
                              </>
                            ) : (
                              <>
                                <div style={{ width: 36, height: 36, borderRadius: '50%', background: '#e5e7eb', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                                  <span style={{ fontSize: 14, color: '#9ca3af' }}>?</span>
                                </div>
                                <span className="text-gray-400 text-sm flex items-center gap-1">
                                  <span className="inline-block w-2 h-2 rounded-full bg-gray-300" style={{ animation: 'pulse-glow 1.5s infinite' }}></span>
                                  Aguardando...
                                </span>
                              </>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>

              {playerSlot === 0 && (
                <button
                  onClick={startGame}
                  disabled={!allHumansJoined}
                  className={'w-full text-lg ' + (allHumansJoined ? 'btn-primary' : '')}
                  style={!allHumansJoined ? { background: '#9ca3af', color: '#e5e7eb', padding: '14px 24px', borderRadius: '12px', fontWeight: 700, fontSize: '1rem', cursor: 'default', border: 'none' } : {}}
                >
                  {!allHumansJoined ? 'Aguardando jogadores...' : 'Comecar Jogo!'}
                </button>
              )}

              {playerSlot !== 0 && (
                <div className="text-center p-4">
                  <p className="text-green-200 font-semibold">Aguardando o host iniciar o jogo...</p>
                </div>
              )}

              {error && (
                <div className="mt-4 p-3 rounded-xl bg-red-500/20 border border-red-400/30 text-red-200 text-center text-sm font-semibold animate-slide-down">
                  {error}
                </div>
              )}
            </div>
          </div>
        );
      }

      // Game Screen
      if (screen === 'game' && gameState) {
        const myHand = gameState.hands?.[playerSlot] || [];
        const topSlot = (playerSlot + 2) % 4;
        const leftSlot = (playerSlot + 1) % 4;
        const rightSlot = (playerSlot + 3) % 4;
        const isMyTurn = gameState.currentPlayer === playerSlot;
        /* Auto-clear side-choice modal when turn changes or tile leaves hand */
        if (choosingTile && (!isMyTurn || !myHand.some(t => t.id === choosingTile.id))) setChoosingTile(null);
        const isBot = (slot) => gameState.players?.[slot] && !gameState.players[slot].isHuman;
        const isPartner = (slot) => slot === topSlot;
        const score0 = gameState.teamScores?.[0] || 0;
        const score1 = gameState.teamScores?.[1] || 0;
        const mt = gameState.matchTarget || 6;

        // Trigger confetti on game end
        if (gameState.gameEnded && !window._confettiFired) {
          window._confettiFired = true;
          setTimeout(launchConfetti, 200);
        }
        if (!gameState.gameEnded) window._confettiFired = false;

        // Opponent tile back renderer
        const TileBack = ({ count, vertical, partner }) => {
          const cls = 'tile-back' + (partner ? ' partner' : '');
          if (vertical) {
            return (
              <div className="flex flex-col gap-0.5 items-center">
                {Array.from({ length: count }).map((_, i) => (
                  <div key={i} className={cls} style={{ width: 10, height: 20 }}></div>
                ))}
              </div>
            );
          }
          return (
            <div className="flex gap-0.5 justify-center flex-wrap">
              {Array.from({ length: count }).map((_, i) => (
                <div key={i} className={cls} style={{ width: 16, height: 28 }}></div>
              ))}
            </div>
          );
        };

        // Score pips renderer
        const ScorePips = ({ score, total, color }) => (
          <div className="flex gap-1.5 items-center">
            {Array.from({ length: total }).map((_, i) => (
              <div
                key={i}
                className={'score-pip' + (i < score ? ' filled' : '')}
                style={{
                  borderColor: color,
                  background: i < score ? color : 'transparent',
                  boxShadow: i < score ? '0 0 6px ' + color + '80' : 'none'
                }}
              />
            ))}
          </div>
        );

        // Compact SVG scoring dial — semicircular arc with both teams
        // Quarter-circle corner dial with needle (like real domino table)
        const CornerDial = ({ score, animScore, total, corner, color, lightColor }) => {
          const S = 74, R = 50, steps = total || 6;
          const isTL = corner === 'tl';
          const ox = isTL ? 2 : S - 2, oy = isTL ? 2 : S - 2;
          const a0 = isTL ? 0 : Math.PI, a1 = isTL ? Math.PI / 2 : Math.PI * 1.5;
          const af = (v) => a0 + (v / steps) * (a1 - a0);
          const needle = animScore != null ? animScore : score;
          const els = [];
          for (let i = 0; i <= steps; i++) {
            const a = af(i), co = Math.cos(a), si = Math.sin(a);
            els.push(<line key={'t'+i} x1={ox+(R-4)*co} y1={oy+(R-4)*si} x2={ox+R*co} y2={oy+R*si} stroke="rgba(255,255,255,0.4)" strokeWidth={1.2} />);
            els.push(<text key={'n'+i} x={ox+(R+9)*co} y={oy+(R+9)*si} fill={i<=score?lightColor:'rgba(255,255,255,0.25)'} fontSize={9} fontWeight={700} fontFamily="monospace" textAnchor="middle" dominantBaseline="central">{i}</text>);
          }
          const bs = {x:ox+R*Math.cos(a0),y:oy+R*Math.sin(a0)}, be = {x:ox+R*Math.cos(a1),y:oy+R*Math.sin(a1)};
          const bgArc = 'M'+bs.x+' '+bs.y+' A'+R+' '+R+' 0 0 1 '+be.x+' '+be.y;
          let scArc = '';
          if (needle > 0) { const se = af(needle); scArc = 'M'+bs.x+' '+bs.y+' A'+R+' '+R+' 0 0 1 '+(ox+R*Math.cos(se))+' '+(oy+R*Math.sin(se)); }
          const na = af(needle), nx = ox+(R-10)*Math.cos(na), ny = oy+(R-10)*Math.sin(na);
          return (
            <svg width={S} height={S} viewBox={'0 0 '+S+' '+S}>
              <path d={bgArc} fill="none" stroke="rgba(255,255,255,0.12)" strokeWidth={2.5} />
              {needle>0 && <path d={scArc} fill="none" stroke={color} strokeWidth={2.5} opacity={0.7} />}
              {els}
              <line x1={ox} y1={oy} x2={nx} y2={ny} stroke={color} strokeWidth={2.5} strokeLinecap="round" />
              <circle cx={ox} cy={oy} r={3} fill={color} stroke="#fff" strokeWidth={1} />
            </svg>
          );
        };

        return (
          <div className="felt-bg animate-fade-in" style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
            <div className="max-w-2xl mx-auto" style={{ width: '100%', flex: 1, display: 'flex', flexDirection: 'column', padding: 8 }}>

              {/* Size + layout toggle */}
              <div className="flex justify-center mb-1 gap-2">
                <div className="inline-flex rounded-md overflow-hidden" style={{ border: '1px solid rgba(255,255,255,0.12)' }}>
                  {['S','M','L'].map(sz => (
                    <button key={sz} onClick={() => setBoardSize(sz)}
                      className="px-1.5 py-0.5 text-[9px] font-bold"
                      style={{ background: boardSize === sz ? 'rgba(255,255,255,0.2)' : 'transparent', color: boardSize === sz ? '#fff' : 'rgba(255,255,255,0.3)' }}>
                      {sz}
                    </button>
                  ))}
                </div>
                <div className="inline-flex rounded-md overflow-hidden" style={{ border: '1px solid rgba(255,255,255,0.12)' }}>
                  {[{k:'spiral',l:'\u25A1'},{k:'snake',l:'\u223F'}].map(ly => (
                    <button key={ly.k} onClick={() => setTileLayout(ly.k)}
                      className="px-1.5 py-0.5 text-[9px] font-bold"
                      style={{ background: tileLayout === ly.k ? 'rgba(255,255,255,0.2)' : 'transparent', color: tileLayout === ly.k ? '#fff' : 'rgba(255,255,255,0.3)' }}>
                      {ly.l}
                    </button>
                  ))}
                </div>
                <button onClick={() => setShowEndBadges(!showEndBadges)}
                  className="px-1.5 py-0.5 text-[9px] font-bold rounded-md"
                  style={{ border: '1px solid rgba(255,255,255,0.12)', background: showEndBadges ? 'rgba(16,185,129,0.3)' : 'transparent', color: showEndBadges ? '#10b981' : 'rgba(255,255,255,0.3)' }}
                  title="Toggle end number badges">
                  🟢
                </button>
                <button onClick={() => setShowStats(true)}
                  className="px-1.5 py-0.5 text-[9px] font-bold rounded-md"
                  style={{ border: '1px solid rgba(255,255,255,0.12)', background: 'transparent', color: 'rgba(255,255,255,0.3)' }}
                  title="Estatísticas">
                  📊
                </button>
              </div>

              {/* Round announcement — rendered below board in the layout flow */}

              {/* Message toast removed — turn info shown via SUA VEZ badge + avatar badges */}

              {/* Starter choice modal — moved to just above player hand */}

              {/* Side choice modal moved to just above player hand */}

              {/* Game Table */}
              <div className="felt-bg rounded-xl p-2 mb-1 relative" style={{ border: '1px solid rgba(255,255,255,0.08)', flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>

                {/* Partner (top) — same team = blue */}
                <div className={'player-panel px-2 py-1.5 mb-1 mx-8 flex items-center justify-center gap-2 flex-wrap ' + (gameState.currentPlayer === topSlot ? 'active-turn turn-pulse' : '')}>
                  <div style={{ position: 'relative', flexShrink: 0, borderRadius: '50%', padding: 2, background: 'rgba(59,130,246,0.7)', boxShadow: '0 0 6px rgba(59,130,246,0.4)' }}>
                    <Avatar profile={profileFromPlayer(gameState.players?.[topSlot])} size={38} noBorder />
                    {!(gameState.currentPlayer === -1 && !gameState.waitingForStarterChoice) && <div style={{ position: 'absolute', bottom: -3, right: -3, width: 16, height: 16, borderRadius: '50%', background: '#16a34a', border: '2px solid #0a2a14', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <span style={{ fontSize: 9, fontWeight: 800, color: '#fff' }}>{(gameState.hands?.[topSlot] || []).length}</span>
                    </div>}
                    {passedSlot === topSlot && <div className="animate-bounce-in" style={{ position: 'absolute', top: -8, left: '50%', transform: 'translateX(-50%)', background: '#ef4444', color: '#fff', fontSize: 9, fontWeight: 800, padding: '2px 6px', borderRadius: 8, whiteSpace: 'nowrap', boxShadow: '0 2px 8px rgba(239,68,68,0.5)' }}>Toquei!</div>}
                  </div>
                  <span className="text-[11px] font-bold text-white/80">{gameState.players?.[topSlot]?.name}</span>
                  {!(gameState.currentPlayer === -1 && !gameState.waitingForStarterChoice) && (isBot(topSlot)
                    ? <span className="text-[9px] px-1 py-0.5 rounded bg-red-500/30 text-red-200 font-bold">BOT</span>
                    : <span className="text-[9px] px-1 py-0.5 rounded bg-green-500/30 text-green-200 font-bold">PARCEIRO</span>
                  )}
                  {gameState.currentPlayer === topSlot && <span className="text-[9px] px-1 py-0.5 rounded bg-yellow-500/30 text-yellow-200 font-bold">VEZ</span>}
                  {!(gameState.currentPlayer === -1 && !gameState.waitingForStarterChoice) ? (
                    <div className="flex gap-0.5 ml-1">
                      {Array.from({ length: (gameState.hands?.[topSlot] || []).length }).map((_, i) => (
                        <div key={i} className="tile-back" style={{ width: 10, height: 18 }}></div>
                      ))}
                    </div>
                  ) : (gameState.hands?.[topSlot] || []).length > 0 ? (
                    <div className="flex gap-0.5 ml-1 flex-wrap">
                      {(gameState.hands?.[topSlot] || []).map(tile => (
                        <BoardTile key={tile.id} tile={tile} orientation="vertical" flipped={false} />
                      ))}
                      {gameState.blockedReveal && (() => {
                        const p = gameState.blockedReveal.players?.find(x => x.slot === topSlot);
                        return p ? <span className="text-[14px] font-extrabold text-yellow-300/90 ml-1">({p.pips})</span> : null;
                      })()}
                    </div>
                  ) : (
                    <span className="text-[9px] text-green-400 font-bold ml-1">Bateu!</span>
                  )}
                </div>

                {/* Middle row */}
                <div className="flex items-stretch mb-1 gap-1" style={{ flex: 1, minHeight: 0 }}>

                  {/* Left opponent — opposing team = red */}
                  <div className={'player-panel p-1 flex-shrink-0 flex flex-col items-center justify-center ' + (gameState.currentPlayer === leftSlot ? 'active-turn turn-pulse' : '')} style={{ width: gameState.currentPlayer === -1 && !gameState.waitingForStarterChoice && (gameState.hands?.[leftSlot] || []).length > 0 ? 64 : 48 }}>
                    <div style={{ position: 'relative', borderRadius: '50%', padding: 2, background: 'rgba(239,68,68,0.7)', boxShadow: '0 0 6px rgba(239,68,68,0.4)' }}>
                      <Avatar profile={profileFromPlayer(gameState.players?.[leftSlot])} size={38} noBorder />
                      {!(gameState.currentPlayer === -1 && !gameState.waitingForStarterChoice) && <div style={{ position: 'absolute', bottom: -4, right: -4, width: 20, height: 20, borderRadius: '50%', background: '#16a34a', border: '2px solid #0a2a14', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <span style={{ fontSize: 11, fontWeight: 800, color: '#fff' }}>{(gameState.hands?.[leftSlot] || []).length}</span>
                      </div>}
                      {passedSlot === leftSlot && <div className="animate-bounce-in" style={{ position: 'absolute', top: -8, left: '50%', transform: 'translateX(-50%)', background: '#ef4444', color: '#fff', fontSize: 9, fontWeight: 800, padding: '2px 6px', borderRadius: 8, whiteSpace: 'nowrap', boxShadow: '0 2px 8px rgba(239,68,68,0.5)' }}>Toquei!</div>}
                    </div>
                    <div className="text-[8px] text-white/60 mt-0.5 text-center truncate font-bold" style={{ maxWidth: 60 }}>{gameState.players?.[leftSlot]?.name}</div>
                    {gameState.currentPlayer === leftSlot && <div className="text-[7px] px-1 py-0.5 rounded bg-yellow-500/30 text-yellow-200 font-bold mt-0.5">VEZ</div>}
                    {!(gameState.currentPlayer === -1 && !gameState.waitingForStarterChoice) ? (
                      <div className="flex flex-col gap-0.5 items-center mt-1">
                        {Array.from({ length: (gameState.hands?.[leftSlot] || []).length }).map((_, i) => (
                          <div key={i} className="tile-back" style={{ width: 14, height: 8 }}></div>
                        ))}
                      </div>
                    ) : (gameState.hands?.[leftSlot] || []).length > 0 ? (
                      <div className="flex flex-wrap gap-0.5 justify-center mt-1">
                        {(gameState.hands?.[leftSlot] || []).map(tile => (
                          <BoardTile key={tile.id} tile={tile} orientation="vertical" flipped={false} />
                        ))}
                        {gameState.blockedReveal && (() => {
                          const p = gameState.blockedReveal.players?.find(x => x.slot === leftSlot);
                          return p ? <div className="text-[14px] font-extrabold text-yellow-300/90 w-full text-center">({p.pips})</div> : null;
                        })()}
                      </div>
                    ) : (
                      <span className="text-[8px] text-green-400 font-bold mt-1">Bateu!</span>
                    )}
                  </div>

                  {/* Board */}
                  <div ref={boardRef} className="board-area flex flex-col items-center justify-center" style={{ position: 'relative', flex: 1, minHeight: 0, overflow: 'hidden', height: boardBox.h ? boardBox.h : 'auto' }}>
                    {(!gameState.board || gameState.board.length === 0) ? null : (() => {
                      const board = gameState.board;
                      const HW = bDims.hw, VW = bDims.vw;
                      const GAP = 1;

                      // End badge (green number)
                      const BADGE_SZ = 30;
                      const EndBadge = ({ num }) => (
                        <div style={{
                          display:'inline-flex', alignItems:'center', justifyContent:'center',
                          width: BADGE_SZ, height: BADGE_SZ, borderRadius: BADGE_SZ / 2,
                          background:'#10b981', border:'2.5px solid #064e3b',
                          boxShadow:'0 0 14px rgba(16,185,129,0.7), 0 0 4px rgba(0,0,0,0.4)',
                          color:'#fff', fontWeight:900, fontSize:15, flexShrink:0,
                          zIndex: 25
                        }}>{num}</div>
                      );

                      // === SPIRAL / RECTANGULAR LAYOUT (default) ===
                      if (tileLayout === 'spiral') {
                        const containerW = (boardBox.w || 300);
                        const containerH = (boardBox.h || 300);

                        // Use the full board area; layoutSpiral avoids dial corners
                        const DIAL_CORNER = 78;
                        const W = Math.max(120, containerW);
                        const H = Math.max(120, containerH);
                        const SPIRAL_PADS = { top: 2, right: 4, bottom: 4, left: 2 };

                        // Auto-scale: try current size, shrink until all tiles fit
                        let spiralHW = HW, spiralVW = VW, out;
                        for (let attempt = 0; attempt < 8; attempt++) {
                          out = layoutSpiral(board, W, H, spiralHW, spiralVW, 3,
                            SPIRAL_PADS, DIAL_CORNER, DIAL_CORNER);
                          if (out.tiles.length >= board.length) break;
                          spiralHW = Math.max(24, Math.round(spiralHW * 0.88));
                          spiralVW = Math.max(12, Math.round(spiralVW * 0.88));
                        }
                        const tilesOut = out.tiles;
                        const first = tilesOut[0];
                        const last = tilesOut[tilesOut.length - 1];

                        return (
                          <div style={{ position:'relative', width:'100%', height:'100%' }}>
                            {tilesOut.map((p, idx) => {
                              const t = board[p.i];
                              const isLeftEnd = idx === 0;
                              const isRightEnd = idx === tilesOut.length - 1;
                              const isEndTile = isLeftEnd || isRightEnd;
                              const isLastPlayed = p.i === board.length - 1 && board.length > 1;
                              // Doubles (spinners) always display horizontally to save space
                              const isDouble = t.left === t.right;
                              const effectiveOrient = isDouble ? 'horizontal' : p.orient;
                              const tw = effectiveOrient === 'horizontal' ? spiralHW : spiralVW;
                              const th = effectiveOrient === 'horizontal' ? spiralVW : spiralHW;
                              return (
                                <div
                                  key={t.id + '-' + p.i}
                                  ref={(el) => { if (el) tileElRef.current.set(t.id + '-' + p.i, el); }}
                                  style={{
                                    position:'absolute', left: p.x, top: p.y,
                                    width: tw, height: th, borderRadius: 4,
                                    zIndex: isLastPlayed ? 5 : isEndTile ? 4 : 3,
                                    boxShadow: isLastPlayed ? '0 0 8px 3px #FFD700' : 'none'
                                  }}
                                >
                                  <BoardTile tile={t} orientation={effectiveOrient} flipped={isDouble ? false : p.flip}
                                    hw={spiralHW} vw={spiralVW}
                                    extraStyle={isEndTile ? {
                                      animation: 'pulse-glow 1.8s ease-in-out infinite',
                                      borderRadius: 4
                                    } : null}
                                  />
                                </div>
                              );
                            })}
                            {/* End badges — optional, fixed to bottom-center of board */}
                            {showEndBadges && (
                              <div style={{
                                position:'absolute', bottom: 4, left: '50%', transform: 'translateX(-50%)',
                                display:'flex', gap: 12, alignItems:'center', zIndex: 25,
                                background:'rgba(0,0,0,0.5)', borderRadius: 16, padding:'3px 10px'
                              }}>
                                <EndBadge num={gameState.leftEnd} />
                                <span style={{color:'rgba(255,255,255,0.4)',fontSize:10}}>•••</span>
                                <EndBadge num={gameState.rightEnd} />
                              </div>
                            )}
                          </div>
                        );
                      }

                      // === HORIZONTAL SNAKE (fallback) ===
                      const containerW = boardBox.w || 300;
                      const maxW = Math.max(160, containerW - 48);
                      const rows = [];
                      let cur = [], w = 0;
                      for (let i = 0; i < board.length; i++) {
                        const t = board[i];
                        const isD = t.left === t.right;
                        const tw = HW; // doubles now horizontal too — all tiles same width
                        const need = cur.length > 0 ? (tw + GAP) : tw;
                        if (w + need > maxW && cur.length > 0) {
                          rows.push(cur); cur = [i]; w = tw;
                        } else { cur.push(i); w += need; }
                      }
                      if (cur.length) rows.push(cur);

                      const joints = [];
                      for (let r = 0; r < rows.length - 1; r++) {
                        const rowA = rows[r], rowB = rows[r + 1];
                        if (!rowA.length || !rowB.length) continue;
                        const oddA = (r % 2) === 1;
                        const endA = oddA ? rowA[0] : rowA[rowA.length - 1];
                        const startB = ((r + 1) % 2) === 1 ? rowB[rowB.length - 1] : rowB[0];
                        joints.push({ endA, startB, r });
                      }

                      return (
                        <div style={{ width:'100%', height:'100%', display:'flex', flexDirection:'column', justifyContent:'center', position:'relative' }}>
                          {rows.map((rowIdxs, rIdx) => {
                            const isOdd = (rIdx % 2) === 1;
                            let flexDir = 'row', justify = 'flex-start';
                            if (rIdx === 0) justify = board.length < 8 ? 'center' : 'flex-end';
                            else if (isOdd) { flexDir = 'row-reverse'; justify = 'flex-start'; }
                            return (
                              <div key={'row'+rIdx} style={{ display:'flex', flexDirection:flexDir, justifyContent:justify, alignItems:'center', gap:GAP, marginBottom: rIdx < rows.length - 1 ? GAP : 0 }}>
                                {rIdx === 0 && showEndBadges && <EndBadge num={gameState.leftEnd} />}
                                {rowIdxs.map((tileIndex) => {
                                  const t = board[tileIndex];
                                  const isD = t.left === t.right;
                                  const orient = 'horizontal'; // doubles displayed horizontally like spinners
                                  const flip = isOdd;
                                  const isLastPlayed = tileIndex === board.length - 1 && board.length > 1;
                                  return (
                                    <div key={t.id + '-' + tileIndex}
                                      ref={(el) => { if (el) tileElRef.current.set(t.id + '-' + tileIndex, el); }}
                                      style={{ display:'inline-block', boxShadow: isLastPlayed ? '0 0 8px 3px #FFD700' : 'none', borderRadius: 4, position:'relative', zIndex: isLastPlayed ? 3 : 'auto' }}>
                                      <BoardTile tile={t} orientation={orient} flipped={flip} />
                                    </div>
                                  );
                                })}
                                {rIdx === rows.length - 1 && showEndBadges && <EndBadge num={gameState.rightEnd} />}
                              </div>
                            );
                          })}
                          <svg style={{ position:'absolute', inset:0, pointerEvents:'none', zIndex:2 }} width="100%" height="100%">
                            {joints.map((j, idx) => {
                              const keyA = board[j.endA].id + '-' + j.endA;
                              const keyB = board[j.startB].id + '-' + j.startB;
                              const elA = tileElRef.current.get(keyA);
                              const elB = tileElRef.current.get(keyB);
                              if (!elA || !elB || !boardRef.current) return null;
                              const ra = elA.getBoundingClientRect();
                              const r0 = boardRef.current.getBoundingClientRect();
                              const oddA = (j.r % 2) === 1;
                              const x1 = (oddA ? ra.left : ra.right) - r0.left;
                              const y1 = ra.bottom - r0.top;
                              return <circle key={idx} cx={x1} cy={y1 + 2} r="3" fill="rgba(245,240,225,0.35)" />;
                            })}
                          </svg>
                        </div>
                      );
                    })()}
                    {/* Dormidas badge — tiny corner indicator */}
                    {gameState.dormidas && gameState.currentPlayer !== -1 && (
                      <div style={{ position: 'absolute', bottom: 4, left: 6, display: 'flex', alignItems: 'center', gap: 3, opacity: 0.4 }}>
                        <div style={{ display: 'flex', gap: 2 }}>
                          {[0,1,2,3].map(i => (
                            <div key={i} style={{ width: bDims.vw, height: bDims.hw, borderRadius: 2, background: '#5a4a3a', border: '1px solid rgba(255,255,255,0.15)' }} />
                          ))}
                        </div>
                      </div>
                    )}
                    {/* Flying tile animation */}
                    {flyingTile && (() => {
                      const animName = flyingTile.fromSlot === playerSlot ? 'fly-from-bottom'
                        : flyingTile.fromSlot === topSlot ? 'fly-from-top'
                        : flyingTile.fromSlot === leftSlot ? 'fly-from-left'
                        : 'fly-from-right';
                      const posStyle = { top: '50%', left: '50%', transform: 'translate(-50%, -50%)' };
                      return (
                        <div style={{
                          position: 'absolute',
                          ...posStyle,
                          zIndex: 50,
                          animation: animName + ' 300ms ease-in-out forwards',
                          pointerEvents: 'none'
                        }}>
                          <BoardTile tile={flyingTile.tile} orientation="vertical" flipped={false} />
                        </div>
                      );
                    })()}
                    {/* Corner score dials — overlaid on board */}
                    <div style={{
                      position: 'absolute', top: 0, left: 0, zIndex: dialPulse === 'team0' ? 30 : 10, pointerEvents: 'none',
                      opacity: dialPulse === 'team0' ? 1 : 0.85,
                      transform: dialPulse === 'team0' ? 'scale(1.15)' : 'scale(1)',
                      transformOrigin: 'top left',
                      transition: 'transform 0.5s ease-out, opacity 0.3s',
                      filter: dialPulse === 'team0' ? 'drop-shadow(0 0 12px rgba(59,130,246,0.7))' : 'none'
                    }}>
                      <CornerDial score={score0} animScore={animScore0} total={mt} corner="tl" color="#3b82f6" lightColor="#93c5fd" />
                    </div>
                    <div style={{
                      position: 'absolute', bottom: 0, right: 0, zIndex: dialPulse === 'team1' ? 30 : 10, pointerEvents: 'none',
                      opacity: dialPulse === 'team1' ? 1 : 0.85,
                      transform: dialPulse === 'team1' ? 'scale(1.15)' : 'scale(1)',
                      transformOrigin: 'bottom right',
                      transition: 'transform 0.5s ease-out, opacity 0.3s',
                      filter: dialPulse === 'team1' ? 'drop-shadow(0 0 12px rgba(239,68,68,0.7))' : 'none'
                    }}>
                      <CornerDial score={score1} animScore={animScore1} total={mt} corner="br" color="#ef4444" lightColor="#fca5a5" />
                    </div>
                    {/* Proxima Rodada — overlays on board after 10s */}
                    {showNextBtn && gameState.currentPlayer === -1 && !gameState.gameEnded && !gameState.waitingForStarterChoice && playerSlot === 0 && (
                      <div style={{ position: 'absolute', bottom: 12, left: '50%', transform: 'translateX(-50%)', zIndex: 40 }}>
                        <button onClick={() => { setRoundCountdown(null); setShowNextBtn(false); newRound(); }}
                          className="btn-primary animate-bounce-in"
                          style={{ fontSize: 15, fontWeight: 800, padding: '10px 28px', whiteSpace: 'nowrap', boxShadow: '0 4px 24px rgba(0,0,0,0.6)', borderRadius: 12 }}>
                          Proxima Rodada{roundCountdown ? ' (' + roundCountdown + 's)' : ''}
                        </button>
                      </div>
                    )}
                  </div>

                  {/* Right opponent — opposing team = red */}
                  <div className={'player-panel p-1 flex-shrink-0 flex flex-col items-center justify-center ' + (gameState.currentPlayer === rightSlot ? 'active-turn turn-pulse' : '')} style={{ width: gameState.currentPlayer === -1 && !gameState.waitingForStarterChoice && (gameState.hands?.[rightSlot] || []).length > 0 ? 64 : 48 }}>
                    <div style={{ position: 'relative', borderRadius: '50%', padding: 2, background: 'rgba(239,68,68,0.7)', boxShadow: '0 0 6px rgba(239,68,68,0.4)' }}>
                      <Avatar profile={profileFromPlayer(gameState.players?.[rightSlot])} size={38} noBorder />
                      {!(gameState.currentPlayer === -1 && !gameState.waitingForStarterChoice) && <div style={{ position: 'absolute', bottom: -4, right: -4, width: 20, height: 20, borderRadius: '50%', background: '#16a34a', border: '2px solid #0a2a14', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <span style={{ fontSize: 11, fontWeight: 800, color: '#fff' }}>{(gameState.hands?.[rightSlot] || []).length}</span>
                      </div>}
                      {passedSlot === rightSlot && <div className="animate-bounce-in" style={{ position: 'absolute', top: -8, left: '50%', transform: 'translateX(-50%)', background: '#ef4444', color: '#fff', fontSize: 9, fontWeight: 800, padding: '2px 6px', borderRadius: 8, whiteSpace: 'nowrap', boxShadow: '0 2px 8px rgba(239,68,68,0.5)' }}>Toquei!</div>}
                    </div>
                    <div className="text-[8px] text-white/60 mt-0.5 text-center truncate font-bold" style={{ maxWidth: 60 }}>{gameState.players?.[rightSlot]?.name}</div>
                    {gameState.currentPlayer === rightSlot && <div className="text-[7px] px-1 py-0.5 rounded bg-yellow-500/30 text-yellow-200 font-bold mt-0.5">VEZ</div>}
                    {!(gameState.currentPlayer === -1 && !gameState.waitingForStarterChoice) ? (
                      <div className="flex flex-col gap-0.5 items-center mt-1">
                        {Array.from({ length: (gameState.hands?.[rightSlot] || []).length }).map((_, i) => (
                          <div key={i} className="tile-back" style={{ width: 14, height: 8 }}></div>
                        ))}
                      </div>
                    ) : (gameState.hands?.[rightSlot] || []).length > 0 ? (
                      <div className="flex flex-wrap gap-0.5 justify-center mt-1">
                        {(gameState.hands?.[rightSlot] || []).map(tile => (
                          <BoardTile key={tile.id} tile={tile} orientation="vertical" flipped={false} />
                        ))}
                        {gameState.blockedReveal && (() => {
                          const p = gameState.blockedReveal.players?.find(x => x.slot === rightSlot);
                          return p ? <div className="text-[14px] font-extrabold text-yellow-300/90 w-full text-center">({p.pips})</div> : null;
                        })()}
                      </div>
                    ) : (
                      <span className="text-[8px] text-green-400 font-bold mt-1">Bateu!</span>
                    )}
                  </div>
                </div>

                {/* Score dials removed from here — now overlaid on board corners */}

                {/* Round announcement banner — below board, above hand */}
                {roundAnnouncement && (
                  <div className="animate-bounce-in" style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10,
                    padding: '6px 16px', margin: '4px 8px 0',
                    background: 'rgba(0,0,0,0.7)', borderRadius: 12,
                    backdropFilter: 'blur(6px)', border: '1px solid rgba(251,191,36,0.3)'
                  }}>
                    <span style={{ fontSize: 28 }}>{roundAnnouncement.emoji}</span>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: 22, fontWeight: 900, color: '#fbbf24', textShadow: '0 2px 6px rgba(0,0,0,0.5)', letterSpacing: 2 }}>
                        {roundAnnouncement.label}
                      </div>
                      <div style={{ fontSize: 12, color: '#fff', fontWeight: 700 }}>
                        {roundAnnouncement.playerName} — +{roundAnnouncement.points} ponto{roundAnnouncement.points > 1 ? 's' : ''}
                      </div>
                    </div>
                  </div>
                )}

                {/* Starter choice — just above player hand */}
                {showStarterChoice && (() => {
                  const winTeam = gameState.lastWinningTeam;
                  const winSlots = winTeam === 0 ? [0, 2] : [1, 3];
                  const isOnWinTeam = winSlots.includes(playerSlot);
                  const isHumanOnWinTeam = gameState.players?.[playerSlot]?.isHuman && isOnWinTeam;
                  if (!isHumanOnWinTeam) return null;
                  const partnerOnTeam = winSlots.find(s => s !== playerSlot);
                  const partnerIsHuman = gameState.players?.[partnerOnTeam]?.isHuman;
                  const myVote = gameState.starterVotes?.[playerSlot];
                  const partnerVote = gameState.starterVotes?.[partnerOnTeam];
                  const hasVoted = myVote !== undefined && myVote !== null;

                  if (!partnerIsHuman) {
                    return (
                      <div className="animate-bounce-in mx-2" style={{
                        background: 'rgba(0,0,0,0.5)', borderRadius: 10, padding: '6px 12px', marginBottom: 4,
                        backdropFilter: 'blur(4px)', border: '1px solid rgba(255,255,255,0.12)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, flexWrap: 'wrap'
                      }}>
                        <span className="text-[11px] font-bold text-white/80">Quem comeca?</span>
                        {starterCountdown !== null && <span className="text-[10px] font-bold text-yellow-300/80">({starterCountdown}s)</span>}
                        <button onClick={() => startGameWithStarter(playerSlot)} className="side-btn right" style={{ padding: '4px 10px', fontSize: 11 }}>Eu</button>
                        <button onClick={() => startGameWithStarter(partnerOnTeam)} className="side-btn left" style={{ padding: '4px 10px', fontSize: 11 }}>{gameState.players[partnerOnTeam]?.name}</button>
                      </div>
                    );
                  }

                  return (
                    <div className="animate-bounce-in mx-2" style={{
                      background: 'rgba(0,0,0,0.5)', borderRadius: 10, padding: '6px 12px', marginBottom: 4,
                      backdropFilter: 'blur(4px)', border: '1px solid rgba(255,255,255,0.12)',
                      display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, flexWrap: 'wrap'
                    }}>
                      {!hasVoted ? (
                        <React.Fragment>
                          <span className="text-[11px] font-bold text-white/80">Quem comeca?</span>
                          {starterCountdown !== null && <span className="text-[10px] font-bold text-yellow-300/80">({starterCountdown}s)</span>}
                          <button onClick={() => submitStarterVote(playerSlot)} className="side-btn right" style={{ padding: '4px 10px', fontSize: 11 }}>Eu</button>
                          <button onClick={() => submitStarterVote(partnerOnTeam)} className="side-btn left" style={{ padding: '4px 10px', fontSize: 11 }}>{gameState.players[partnerOnTeam]?.name}</button>
                        </React.Fragment>
                      ) : (
                        <React.Fragment>
                          <span className="text-[11px] font-bold text-green-400">Voto registrado!</span>
                          <span className="text-[10px] text-white/50">
                            {partnerVote !== undefined && partnerVote !== null ? 'Decidindo...' : 'Esperando ' + gameState.players[partnerOnTeam]?.name + '...'}
                          </span>
                          {starterCountdown !== null && <span className="text-[10px] font-bold text-yellow-300/80">({starterCountdown}s)</span>}
                        </React.Fragment>
                      )}
                    </div>
                  );
                })()}

                {/* Side choice modal — just above player hand */}
                {choosingTile && isMyTurn && myHand.some(t => t.id === choosingTile.id) && (
                  <div className="animate-bounce-in mx-2" style={{
                    background: 'rgba(0,0,0,0.5)', borderRadius: 10, padding: '6px 12px', marginBottom: 4,
                    backdropFilter: 'blur(4px)', border: '1px solid rgba(255,255,255,0.12)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8
                  }}>
                    <span className="text-[11px] font-bold text-white/80">{choosingTile.left}-{choosingTile.right}:</span>
                    <button onClick={() => { playTile(choosingTile, 'left'); setChoosingTile(null); }} className="side-btn left" style={{ padding: '4px 10px', fontSize: 11 }}>Esq ({gameState.leftEnd})</button>
                    <button onClick={() => { playTile(choosingTile, 'right'); setChoosingTile(null); }} className="side-btn right" style={{ padding: '4px 10px', fontSize: 11 }}>Dir ({gameState.rightEnd})</button>
                    <button onClick={() => setChoosingTile(null)} className="text-white/30 text-[10px] hover:text-white/60">✕</button>
                  </div>
                )}

                {/* My Hand — my team = blue */}
                <div className={'my-hand px-1.5 py-1 mx-2 ' + (isMyTurn ? 'my-turn' : '')} style={{ marginTop: 4 }}>
                  <div className="flex items-center justify-center gap-1.5" style={{ marginBottom: 2 }}>
                    <div style={{ position: 'relative', flexShrink: 0, borderRadius: '50%', padding: 2, background: 'rgba(59,130,246,0.7)', boxShadow: '0 0 6px rgba(59,130,246,0.4)' }}>
                      <Avatar profile={profileFromPlayer(gameState.players?.[playerSlot])} size={38} noBorder />
                      <div style={{ position: 'absolute', bottom: -2, right: -2, width: 14, height: 14, borderRadius: '50%', background: '#16a34a', border: '1.5px solid #0a2a14', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <span style={{ fontSize: 8, fontWeight: 800, color: '#fff' }}>{myHand.length}</span>
                      </div>
                      {passedSlot === playerSlot && <div className="animate-bounce-in" style={{ position: 'absolute', top: -8, left: '50%', transform: 'translateX(-50%)', background: '#ef4444', color: '#fff', fontSize: 9, fontWeight: 800, padding: '2px 6px', borderRadius: 8, whiteSpace: 'nowrap', boxShadow: '0 2px 8px rgba(239,68,68,0.5)' }}>Toquei!</div>}
                    </div>
                    <span className="text-[10px] font-bold text-white/80">Voce</span>
                    {isMyTurn && <span className="text-[8px] px-1 py-0.5 rounded-full font-bold turn-pulse" style={{ background: 'rgba(250,204,21,0.25)', color: '#fde047' }}>SUA VEZ</span>}
                    {gameState.currentPlayer === -1 && gameState.blockedReveal && (() => {
                      const p = gameState.blockedReveal.players?.find(x => x.slot === playerSlot);
                      return p ? <span className="text-[14px] font-extrabold text-yellow-300/90">({p.pips})</span> : null;
                    })()}
                  </div>

                  <div className="flex flex-wrap gap-1 justify-center" style={{ marginBottom: 2 }}>
                    {myHand.map(tile => (
                      <DominoTile
                        key={tile.id}
                        tile={tile}
                        playable={isMyTurn && canPlayTile(tile)}
                        hDims={hDims}
                        onClick={() => {
                          if (!isMyTurn || !canPlayTile(tile)) return;
                          playTile(tile);
                        }}
                      />
                    ))}
                  </div>
                  {isMyTurn && !myHand.some(t => canPlayTile(t)) && (
                    <div className="text-center text-[10px] font-bold text-yellow-300/80 animate-slide-down">
                      Sem peca — passando...
                    </div>
                  )}
                  {isMyTurn && moveTimer !== null && myHand.some(t => canPlayTile(t)) && (
                    <div className="text-center">
                      <span className={'inline-block px-1.5 py-0 rounded-full text-[9px] font-bold ' + (moveTimer <= 3 ? 'bg-red-500/30 text-red-300' : 'bg-white/10 text-white/60')}>
                        {moveTimer}s
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* Dormidas indicator — subtle inline during play */}

              {/* Dormidas revealed after round ends */}
              {gameState.currentPlayer === -1 && !gameState.waitingForStarterChoice && gameState.dormidas && gameState.dormidas.length > 0 && (
                <div className="px-2 py-1.5 rounded-lg mb-1" style={{ background: 'rgba(0,0,0,0.15)', border: '1px solid rgba(255,255,255,0.06)' }}>

                  <div className="flex gap-1 justify-center flex-wrap">
                    {gameState.dormidas.map(tile => (
                      <BoardTile key={tile.id} tile={tile} orientation="vertical" flipped={false} />
                    ))}
                  </div>
                </div>
              )}

              {/* Blocked game reveal — show all players' remaining tiles face-up */}
              {gameState.currentPlayer === -1 && !gameState.waitingForStarterChoice && gameState.blockedReveal && (
                <div className="rounded-xl mb-1 animate-bounce-in" style={{
                  background: 'rgba(0,0,0,0.35)',
                  border: '1px solid rgba(255,255,255,0.12)',
                  padding: '8px 10px',
                  backdropFilter: 'blur(4px)'
                }}>
                  {/* Header */}
                  <div className="text-center mb-2">
                    <div className="text-sm font-extrabold" style={{ color: '#fbbf24', textShadow: '0 0 10px rgba(251,191,36,0.4)' }}>
                      Jogo Trancado!
                    </div>
                    {gameState.blockedReveal.isDobrada ? (
                      <div className="text-[10px] text-white/60 font-bold">Empate — dobrada!</div>
                    ) : (
                      <div className="text-[10px] text-white/60 font-bold">
                        {gameState.players[gameState.blockedReveal.winnerSlot]?.name} venceu com menos pontos
                      </div>
                    )}
                  </div>

                  {/* Individual player summaries — jogo trancado is won by the player with fewest pips */}
                  <div className="grid grid-cols-2 gap-1 mb-2">
                    {gameState.blockedReveal.players?.slice().sort((a, b) => a.pips - b.pips).map(p => {
                      const isWinner = !gameState.blockedReveal.isDobrada && gameState.blockedReveal.winnerSlot === p.slot;
                      const isTie = gameState.blockedReveal.isDobrada;
                      const teamColor = p.slot % 2 === 0 ? '#3b82f6' : '#ef4444';
                      return (
                        <div key={p.slot} className="rounded-lg" style={{
                          background: isWinner ? 'rgba(34,197,94,0.15)' : isTie ? 'rgba(251,191,36,0.1)' : 'rgba(255,255,255,0.04)',
                          border: '1px solid ' + (isWinner ? 'rgba(34,197,94,0.35)' : isTie ? 'rgba(251,191,36,0.2)' : 'rgba(255,255,255,0.08)'),
                          padding: '4px 6px'
                        }}>
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-[9px] font-bold truncate mr-1" style={{ color: teamColor }}>{p.name}</span>
                            <span className={'text-[11px] font-extrabold ' + (isWinner ? 'text-green-400' : isTie ? 'text-yellow-400' : 'text-white/50')}>
                              {p.pips} pts
                            </span>
                          </div>
                          {isWinner && <div className="text-[8px] text-green-400/80 font-bold text-center">VENCEU</div>}
                        </div>
                      );
                    })}
                  </div>

                  {/* Tiles shown inline next to each player's avatar */}
                </div>
              )}

              {/* Action button removed from here — now overlays on board after 10s */}

              {gameState.gameEnded && (() => {
                const myTeam = playerSlot % 2;
                const won = (gameState.teamScores?.[myTeam] || 0) >= (gameState.matchTarget || 6);
                const s0 = gameState.teamScores?.[0] || 0;
                const s1 = gameState.teamScores?.[1] || 0;
                const isBuchuda = (s0 === 0 || s1 === 0) && (s0 >= (gameState.matchTarget || 6) || s1 >= (gameState.matchTarget || 6));
                const buchuWord = Math.random() < 0.5 ? 'BUCHUDA' : 'DEDADA';
                const borderColor = won ? 'rgba(250,204,21,0.5)' : 'rgba(239,68,68,0.5)';
                return (
                  <div className="animate-bounce-in rounded-xl mx-2 mt-1" style={{
                    background: 'rgba(30,40,60,0.95)', border: '2px solid ' + borderColor,
                    boxShadow: '0 4px 20px rgba(0,0,0,0.4)', padding: '10px 16px'
                  }}>
                    <div className="flex items-center gap-3">
                      <div className="text-3xl">{won ? (isBuchuda ? '\uD83D\uDCA5' : '\uD83C\uDFC6') : (isBuchuda ? '\uD83D\uDCA9' : '\uD83D\uDC80')}</div>
                      <div className="flex-1">
                        {isBuchuda && (
                          <div className="text-lg font-black" style={{ color: '#f59e0b', textShadow: '0 0 12px rgba(245,158,11,0.5)', letterSpacing: 2 }}>
                            {buchuWord}!
                          </div>
                        )}
                        <div className={'text-base font-extrabold ' + (won ? 'text-yellow-400' : 'text-red-400')}>
                          {won ? 'PARTIDA GANHA!' : 'PARTIDA PERDIDA!'}
                        </div>
                        <div className="text-white/60 text-[11px]">
                          {isBuchuda
                            ? (won ? 'Venceu sem deixar ponto!' : 'Perdeu sem marcar!')
                            : (won ? 'Seu time venceu!' : 'Adversarios venceram!')}
                        </div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-extrabold text-white">{s0} - {s1}</div>
                        <div className="text-[9px] text-white/40">a {gameState.matchTarget || 6} pts</div>
                      </div>
                    </div>
                    <div className="flex gap-2 mt-2">
                      <button onClick={startGame} className="btn-primary flex-1 py-2 text-sm font-bold">Nova Partida</button>
                      <button onClick={() => { db.ref('rooms/' + roomCode).remove(); setGameState(null); setRoomCode(''); setScreen('menu'); }} className="flex-1 py-2 text-xs font-bold rounded-xl" style={{ background: 'rgba(255,255,255,0.1)', color: 'rgba(255,255,255,0.7)', border: '1px solid rgba(255,255,255,0.15)' }}>Menu</button>
                    </div>
                  </div>
                );
              })()}

              {error && (
                <div className="mt-2 p-3 rounded-xl bg-red-500/20 border border-red-400/30 text-red-200 text-center text-sm font-semibold animate-slide-down">
                  {error}
                </div>
              )}

              {/* Statistics Modal */}
              {showStats && (() => {
                const stats = playerStatsRef.current;
                const players = gameState.players || {};
                const teamColors = ['#22c55e', '#f59e0b'];
                const teamNames = ['Time 1', 'Time 2'];
                const statEntries = [0, 1, 2, 3].map(i => {
                  const p = players[i];
                  if (!p) return null;
                  return stats[p.name] || { name: p.name, team: i % 2, matchesWon: 0, matchesLost: 0, roundsWon: 0, roundsLost: 0, winTypes: { normal: 0, cruzada: 0, 'com carroca': 0, 'la e lo': 0, blocked: 0 }, buchudaGiven: 0, buchudaReceived: 0, totalPoints: 0 };
                }).filter(Boolean);

                return (
                  <div onClick={() => setShowStats(false)} style={{
                    position: 'fixed', inset: 0, zIndex: 9999,
                    background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(8px)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    padding: 12, animation: 'fade-in 0.2s ease-out'
                  }}>
                    <div onClick={e => e.stopPropagation()} style={{
                      background: 'linear-gradient(145deg, #1e293b, #0f172a)',
                      border: '1px solid rgba(255,255,255,0.15)',
                      borderRadius: 16, padding: 16, width: '100%', maxWidth: 420,
                      maxHeight: '90vh', overflowY: 'auto',
                      boxShadow: '0 8px 32px rgba(0,0,0,0.6)'
                    }}>
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <span style={{ fontSize: 20 }}>📊</span>
                          <span style={{ fontSize: 16, fontWeight: 800, color: '#fff' }}>Estatísticas</span>
                        </div>
                        <button onClick={() => setShowStats(false)} style={{
                          background: 'rgba(255,255,255,0.1)', border: 'none', borderRadius: 8,
                          color: '#fff', fontSize: 16, width: 32, height: 32, cursor: 'pointer',
                          display: 'flex', alignItems: 'center', justifyContent: 'center'
                        }}>✕</button>
                      </div>

                      {statEntries.map((s, idx) => {
                        const teamColor = teamColors[s.team];
                        const totalMatches = s.matchesWon + s.matchesLost;
                        const totalRounds = s.roundsWon + s.roundsLost;
                        const winRate = totalMatches > 0 ? Math.round((s.matchesWon / totalMatches) * 100) : 0;
                        return (
                          <div key={s.name} style={{
                            background: 'rgba(255,255,255,0.05)',
                            border: '1px solid ' + teamColor + '33',
                            borderRadius: 12, padding: 12, marginBottom: idx < statEntries.length - 1 ? 8 : 0
                          }}>
                            <div className="flex items-center gap-2 mb-2">
                              <Avatar profile={profileFromPlayer(players[[0,1,2,3].find(i => players[i]?.name === s.name)])} size={32} noBorder />
                              <div style={{ flex: 1 }}>
                                <div style={{ fontSize: 13, fontWeight: 700, color: '#fff' }}>{s.name}</div>
                                <div style={{ fontSize: 9, fontWeight: 600, color: teamColor }}>{teamNames[s.team]}</div>
                              </div>
                              {totalMatches > 0 && (
                                <div style={{ textAlign: 'center' }}>
                                  <div style={{ fontSize: 18, fontWeight: 800, color: teamColor }}>{winRate}%</div>
                                  <div style={{ fontSize: 8, color: 'rgba(255,255,255,0.4)' }}>vitórias</div>
                                </div>
                              )}
                            </div>

                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 12px', fontSize: 10 }}>
                              <div style={{ color: 'rgba(255,255,255,0.5)' }}>Partidas</div>
                              <div style={{ color: '#fff', fontWeight: 600, textAlign: 'right' }}>
                                <span style={{ color: '#22c55e' }}>{s.matchesWon}W</span>
                                {' / '}
                                <span style={{ color: '#ef4444' }}>{s.matchesLost}L</span>
                              </div>
                              <div style={{ color: 'rgba(255,255,255,0.5)' }}>Rodadas</div>
                              <div style={{ color: '#fff', fontWeight: 600, textAlign: 'right' }}>
                                <span style={{ color: '#22c55e' }}>{s.roundsWon}W</span>
                                {' / '}
                                <span style={{ color: '#ef4444' }}>{s.roundsLost}L</span>
                              </div>
                              <div style={{ color: 'rgba(255,255,255,0.5)' }}>Pontos</div>
                              <div style={{ color: '#fff', fontWeight: 600, textAlign: 'right' }}>{s.totalPoints}</div>
                            </div>

                            <div style={{ marginTop: 6, paddingTop: 6, borderTop: '1px solid rgba(255,255,255,0.08)' }}>
                              <div style={{ fontSize: 9, fontWeight: 700, color: 'rgba(255,255,255,0.4)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: 1 }}>Tipos de vitória</div>
                              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                                {[
                                  { key: 'normal', label: 'Batida', emoji: '✅', color: '#22c55e' },
                                  { key: 'cruzada', label: 'Cruzada', emoji: '💥', color: '#f59e0b' },
                                  { key: 'com carroca', label: 'Carroça', emoji: '🎯', color: '#3b82f6' },
                                  { key: 'la e lo', label: 'Lá e Ló', emoji: '🔥', color: '#ef4444' },
                                  { key: 'blocked', label: 'Travada', emoji: '🔒', color: '#8b5cf6' }
                                ].map(wt => {
                                  const count = s.winTypes[wt.key] || 0;
                                  return (
                                    <div key={wt.key} style={{
                                      background: count > 0 ? wt.color + '20' : 'rgba(255,255,255,0.03)',
                                      border: '1px solid ' + (count > 0 ? wt.color + '40' : 'rgba(255,255,255,0.06)'),
                                      borderRadius: 6, padding: '2px 6px',
                                      fontSize: 9, fontWeight: 600,
                                      color: count > 0 ? wt.color : 'rgba(255,255,255,0.2)'
                                    }}>
                                      {wt.emoji} {wt.label} {count}
                                    </div>
                                  );
                                })}
                              </div>
                            </div>

                            {(s.buchudaGiven > 0 || s.buchudaReceived > 0) && (
                              <div style={{ marginTop: 6, paddingTop: 6, borderTop: '1px solid rgba(255,255,255,0.08)' }}>
                                <div style={{ fontSize: 9, fontWeight: 700, color: 'rgba(255,255,255,0.4)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: 1 }}>Buchuda</div>
                                <div style={{ display: 'flex', gap: 8, fontSize: 10 }}>
                                  {s.buchudaGiven > 0 && (
                                    <span style={{ color: '#f59e0b', fontWeight: 700 }}>💀 Deu: {s.buchudaGiven}</span>
                                  )}
                                  {s.buchudaReceived > 0 && (
                                    <span style={{ color: '#ef4444', fontWeight: 700 }}>💩 Levou: {s.buchudaReceived}</span>
                                  )}
                                </div>
                              </div>
                            )}
                          </div>
                        );
                      })}

                      {statEntries.every(s => s.matchesWon === 0 && s.matchesLost === 0 && s.roundsWon === 0 && s.roundsLost === 0) && (
                        <div style={{ textAlign: 'center', padding: 20, color: 'rgba(255,255,255,0.3)', fontSize: 12 }}>
                          Nenhuma estatística ainda. Jogue uma rodada!
                        </div>
                      )}
                    </div>
                  </div>
                );
              })()}
            </div>
          </div>
        );
      }

      return null;
    }

    // Auto-load neural model if available
    loadNeuralModel('domino_model.bin').catch(() => {
      console.log('Neural model not found — using heuristic AI only');
      USE_NN_LEAF_VALUE = false;
    });

    ReactDOM.createRoot(document.getElementById('root')).render(<App />);
  