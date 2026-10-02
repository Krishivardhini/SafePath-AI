// SafePath Service Worker — Always Network First to ensure latest UI updates
const CACHE_NAME = "safepath-v2.2-multilingual-perm";

self.addEventListener("install", (e) => {
  self.skipWaiting();
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.map((k) => caches.delete(k))
      );
    })
  );
  self.clients.claim();
});

self.addEventListener("fetch", (e) => {
  // Let API, WebSocket, and assets fetch fresh from network
  if (e.request.url.includes("/api/") || e.request.url.includes("/ws/")) {
    return;
  }
  e.respondWith(
    fetch(e.request).catch(() => caches.match(e.request))
  );
});
