const CACHE_NAME = 'keyflow-shell-v4.11';

const PRECACHE_URLS = [
  './',
  './index.html',
  './manifest.json',
  './assets/icon-180.png',
  './assets/icon-512.png',
  './css/styles.css?v=4.11',
  './js/songs.js?v=4.11',
  './js/midi-parser.js?v=4.11',
  './js/audio-transcriber.js?v=4.11',
  './js/audio.js?v=4.11',
  './js/app.js?v=4.11'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(PRECACHE_URLS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(
        keys
          .filter((key) => key.startsWith('keyflow-shell-') && key !== CACHE_NAME)
          .map((key) => caches.delete(key))
      ))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  const scopePath = new URL('./', self.location.href).pathname;
  if (
    url.pathname.startsWith('/api/') ||
    url.pathname === '/health' ||
    url.pathname.startsWith(`${scopePath}api/`) ||
    url.pathname === `${scopePath}health`
  ) {
    return;
  }

  event.respondWith((async () => {
    const cache = await caches.open(CACHE_NAME);
    const cached = await cache.match(request);

    const networkRefresh = fetch(request)
      .then((networkResp) => {
        if (networkResp && networkResp.ok && networkResp.type === 'basic') {
          return cache.put(request, networkResp.clone()).then(() => networkResp);
        }
        return networkResp;
      })
      .catch(() => null);

    if (cached) {
      return cached;
    }

    const networkResp = await networkRefresh;
    if (networkResp) return networkResp;

    if (request.mode === 'navigate') {
      const fallback = (await cache.match('./index.html')) || (await cache.match('./'));
      if (fallback) return fallback;
    }

    return Response.error();
  })());
});
