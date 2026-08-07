/**
 * Polymarket API Proxy — Cloudflare Worker
 * 
 * Proxies requests to Polymarket's Gamma API and CLOB API.
 * Deploy this to bypass ISP blocking in restricted regions.
 * 
 * Routes:
 *   /gamma/*   → gamma-api.polymarket.com/*
 *   /clob/*    → clob.polymarket.com/*
 *   /health    → Health check
 */

// Allowed origin patterns (set to "*" for open access, or restrict to your domain)
const ALLOWED_ORIGINS = "*";

// Rate limiting: max requests per minute per IP
const RATE_LIMIT_MAX = 60;
const RATE_LIMIT_WINDOW = 60; // seconds

// Simple in-memory rate limit store (per worker instance)
const rateLimitStore = new Map();

function cleanupRateLimit() {
  const now = Date.now();
  for (const [key, value] of rateLimitStore.entries()) {
    if (now - value.windowStart > RATE_LIMIT_WINDOW * 1000) {
      rateLimitStore.delete(key);
    }
  }
}

function checkRateLimit(ip) {
  const now = Date.now();
  const entry = rateLimitStore.get(ip);

  if (!entry || now - entry.windowStart > RATE_LIMIT_WINDOW * 1000) {
    rateLimitStore.set(ip, { count: 1, windowStart: now });
    return true;
  }

  if (entry.count >= RATE_LIMIT_MAX) {
    return false;
  }

  entry.count++;
  return true;
}

function corsHeaders() {
  return {
    "Access-Control-Allow-Origin": ALLOWED_ORIGINS,
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
    "Access-Control-Max-Age": "86400",
  };
}

function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json",
      ...corsHeaders(),
    },
  });
}

function errorResponse(message, status = 500) {
  return jsonResponse({ error: message, status }, status);
}

export default {
  async fetch(request, env, ctx) {
    // Handle CORS preflight
    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: corsHeaders(),
      });
    }

    const url = new URL(request.url);
    const path = url.pathname;
    const clientIP = request.headers.get("CF-Connecting-IP") || "unknown";

    // Rate limiting
    if (!checkRateLimit(clientIP)) {
      return errorResponse("Rate limit exceeded. Max 60 requests/minute.", 429);
    }

    // Health check
    if (path === "/health" || path === "/") {
      return jsonResponse({
        status: "ok",
        service: "polymarket-proxy",
        timestamp: new Date().toISOString(),
        routes: ["/gamma/*", "/clob/*", "/health"],
      });
    }

    // Route: /gamma/* → gamma-api.polymarket.com/*
    if (path.startsWith("/gamma/") || path.startsWith("/gamma")) {
      const targetPath = path.replace(/^\/gamma\/?/, "");
      const targetUrl = `https://gamma-api.polymarket.com/${targetPath}${url.search}`;

      try {
        const headers = new Headers(request.headers);
        headers.set("Host", "gamma-api.polymarket.com");
        headers.set("User-Agent", "HKWeatherBet-Proxy/1.0");
        headers.delete("CF-Connecting-IP");
        headers.delete("X-Forwarded-For");

        const response = await fetch(targetUrl, {
          method: request.method,
          headers,
          body: request.method !== "GET" ? request.body : undefined,
        });

        const responseBody = await response.text();

        return new Response(responseBody, {
          status: response.status,
          headers: {
            "Content-Type": response.headers.get("Content-Type") || "application/json",
            ...corsHeaders(),
            "X-Proxy-By": "polymarket-proxy-worker",
          },
        });
      } catch (err) {
        return errorResponse(`Gamma API error: ${err.message}`, 502);
      }
    }

    // Route: /clob/* → clob.polymarket.com/*
    if (path.startsWith("/clob/") || path.startsWith("/clob")) {
      const targetPath = path.replace(/^\/clob\/?/, "");
      const targetUrl = `https://clob.polymarket.com/${targetPath}${url.search}`;

      try {
        const headers = new Headers(request.headers);
        headers.set("Host", "clob.polymarket.com");
        headers.set("User-Agent", "HKWeatherBet-Proxy/1.0");
        headers.delete("CF-Connecting-IP");
        headers.delete("X-Forwarded-For");

        const response = await fetch(targetUrl, {
          method: request.method,
          headers,
          body: request.method !== "GET" ? request.body : undefined,
        });

        const responseBody = await response.text();

        return new Response(responseBody, {
          status: response.status,
          headers: {
            "Content-Type": response.headers.get("Content-Type") || "application/json",
            ...corsHeaders(),
            "X-Proxy-By": "polymarket-proxy-worker",
          },
        });
      } catch (err) {
        return errorResponse(`CLOB API error: ${err.message}`, 502);
      }
    }

    // 404 for unknown routes
    return errorResponse("Not found. Use /gamma/* or /clob/* routes.", 404);
  },
};
