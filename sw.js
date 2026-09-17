const VERSION = 'v2';
const STATIC_CACHE = `meghalaya-mornings-static-${VERSION}`;
const RUNTIME_CACHE = `meghalaya-mornings-runtime-${VERSION}`;
const DATA_CACHE = `meghalaya-mornings-data-${VERSION}`;
const CACHE_NAMES = [STATIC_CACHE, RUNTIME_CACHE, DATA_CACHE];
const APP_SHELL = ['./', './village_simulator.html', './sw.js', './manifest.webmanifest', './village-background.png'];
const SCENARIOS_URL = '/api/v1/cultural/scenario';

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then(async cache => {
        const requiredShell = APP_SHELL.filter(asset => asset !== './village-background.png');
        await cache.addAll(requiredShell);
        await cache.add('./village-background.png').catch(() => {});
      })
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys()
      .then(keys => Promise.all(
        keys
          .filter(key => key.startsWith('meghalaya-mornings-') && !CACHE_NAMES.includes(key))
          .map(key => caches.delete(key))
      ))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', event => {
  const request = event.request;
  if (request.method !== 'GET') return;

  const url = new URL(request.url);
  if (isTelemetryRequest(url) || isAnalyticsRequest(url)) {
    event.respondWith(fetch(request));
    return;
  }

  if (url.origin !== self.location.origin) {
    event.respondWith(fetch(request));
    return;
  }

  if (request.mode === 'navigate') {
    event.respondWith(networkFirstNavigation(request));
    return;
  }

  if (url.pathname === SCENARIOS_URL) {
    event.respondWith(staleWhileRevalidate(request));
    return;
  }

  if (isStaticAsset(url)) event.respondWith(cacheFirst(request));
});

async function cacheFirst(request) {
  const cached = await caches.match(request, { cacheName: STATIC_CACHE });
  if (cached) return cached;
  const response = await fetch(request);
  if (response.ok) {
    const cache = await caches.open(STATIC_CACHE);
    await cache.put(request, response.clone());
  }
  return response;
}

async function networkFirstNavigation(request) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(STATIC_CACHE);
      await cache.put('./village_simulator.html', response.clone());
    }
    return response;
  } catch (error) {
    const cached = await caches.match('./village_simulator.html', { cacheName: STATIC_CACHE });
    return cached || new Response('Offline page unavailable', { status: 503, headers: { 'Content-Type': 'text/plain' } });
  }
}

async function staleWhileRevalidate(request) {
  const cache = await caches.open(DATA_CACHE);
  const cached = await cache.match(request);
  const refresh = fetch(request)
    .then(response => {
      if (!response.ok) return null;
      cache.put(request, response.clone());
      return response;
    })
    .catch(() => null);

  if (cached) return cached;
  const response = await refresh;
  return response || new Response(JSON.stringify({ scenarios: [] }), { status: 503, headers: { 'Content-Type': 'application/json' } });
}

function isStaticAsset(url) {
  const assetPath = url.pathname.toLowerCase();
  const hasStaticExtension = /\.(?:css|js|mjs|woff2?|ttf|otf|eot|png|jpe?g|webp|svg|ico)$/.test(assetPath);
  const hasContentHash = /[._-][a-f0-9]{8,}(?:\.|$)/i.test(assetPath);
  return hasStaticExtension && (hasContentHash || assetPath.startsWith('/assets/'));
}

function isTelemetryRequest(url) { return url.pathname === '/api/v1/telemetry/log' || url.pathname === '/api/v1/ml/analyze'; }
function isAnalyticsRequest(url) { return /(?:analytics|telemetry|google-analytics|doubleclick)/i.test(url.hostname + url.pathname); }

self.addEventListener('sync', event => {
  if (event.tag === 'telemetry-sync') event.waitUntil(syncTelemetry());
  if (event.tag === 'scenario-sync') event.waitUntil(fetchFreshScenarios());
});

async function syncTelemetry() {
  const records = await readTelemetry();
  if (!records.length) return;
  const response = await fetch('/api/v1/ml/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessions: records }),
  });
  if (response.ok) {
    await clearTelemetry();
    return;
  }
  if (isTransientStatus(response.status)) throw new Error(`Telemetry retryable response: ${response.status}`);
  await clearTelemetry();
}

async function fetchFreshScenarios() {
  const response = await fetch(SCENARIOS_URL);
  if (response.ok) {
    const cache = await caches.open(DATA_CACHE);
    await cache.put(SCENARIOS_URL, response.clone());
    return;
  }
  if (isTransientStatus(response.status)) throw new Error(`Scenario retryable response: ${response.status}`);
}

function isTransientStatus(status) { return status === 408 || status === 425 || status === 429 || status >= 500; }

self.addEventListener('push', event => {
  const payload = parsePushPayload(event);
  event.waitUntil(self.registration.showNotification(payload.title, {
    body: payload.body,
    icon: payload.icon,
    badge: payload.badge,
    tag: payload.tag,
    data: { url: payload.url },
  }));
});

self.addEventListener('notificationclick', event => {
  event.notification.close();
  const targetUrl = new URL(event.notification.data?.url || './', self.location.origin).href;
  event.waitUntil(self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then(clients => {
    const existing = clients.find(client => client.url === targetUrl);
    return existing ? existing.focus() : self.clients.openWindow(targetUrl);
  }));
});

function parsePushPayload(event) {
  const fallback = {
    title: 'Meghalaya Mornings',
    body: 'A new village activity is ready.',
    icon: './icon-192.png',
    badge: './icon-192.png',
    tag: 'village-update',
    url: './village_simulator.html',
  };
  if (!event.data) return fallback;
  try { return { ...fallback, ...event.data.json() }; } catch (error) { return { ...fallback, body: event.data.text() }; }
}

function openTelemetryDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('meghalaya-mornings', 1);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function readTelemetry() {
  const db = await openTelemetryDatabase();
  return new Promise((resolve, reject) => {
    const request = db.transaction('telemetry', 'readonly').objectStore('telemetry').getAll();
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function clearTelemetry() {
  const db = await openTelemetryDatabase();
  return new Promise((resolve, reject) => {
    const request = db.transaction('telemetry', 'readwrite').objectStore('telemetry').clear();
    request.onsuccess = resolve;
    request.onerror = () => reject(request.error);
  });
}
