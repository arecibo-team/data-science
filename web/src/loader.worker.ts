// Fetches, gunzips and decodes data files off the main thread.
// Month files are decoded straight into typed arrays, which are transferred (not copied) back.

import type { Columns } from "./types";

interface Request {
  id: number;
  kind: "month" | "json";
  url: string;
}

// Untyped view of the worker global, so this file does not clash with the DOM lib types.
const ctx = self as unknown as {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  onmessage: ((e: MessageEvent<Request>) => void) | null;
};

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Fetch a gzipped file and return its decompressed bytes. Network errors are retried with backoff.
 *
 * Some servers (the Vite dev server among them) send .gz files with `Content-Encoding: gzip`.
 * The browser then decodes the transport layer itself and hands us plain bytes, so the body is
 * only gunzipped here when the gzip magic is still present.
 */
async function fetchGunzipped(url: string, attempts = 4): Promise<ArrayBuffer> {
  for (let i = 0; ; i++) {
    let res: Response;
    try {
      res = await fetch(url);
    } catch (e) {
      if (i + 1 >= attempts) throw e;
      await sleep(250 * 2 ** i);
      continue;
    }
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
    const bytes = new Uint8Array(await res.arrayBuffer());
    if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes.buffer;
    return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
  }
}

/**
 * Decode a month file written by firms/export.py: a uint32 header length, a JSON header,
 * then each column as raw little-endian bytes in a fixed order.
 */
function decodeMonth(buf: ArrayBuffer) {
  const hlen = new DataView(buf).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hlen))) as {
    n: number;
    incidents: string[];
  };
  const n = header.n;
  let off = 4 + hlen;
  const transfer: ArrayBuffer[] = [];
  const take = <T extends { buffer: ArrayBufferLike; BYTES_PER_ELEMENT: number; length: number }>(
    Ctor: new (b: ArrayBuffer) => T,
  ): T => {
    const size = Ctor.prototype.BYTES_PER_ELEMENT * n;
    const arr = new Ctor(buf.slice(off, off + size));
    off += size;
    transfer.push(arr.buffer as ArrayBuffer);
    return arr;
  };
  // Must match COLUMNS in firms/export.py.
  const cols: Columns = {
    lat: take(Float32Array),
    lon: take(Float32Array),
    frp: take(Float32Array),
    bt: take(Float32Array),
    cluster: take(Int32Array),
    minute: take(Uint16Array),
    incident: take(Uint16Array),
    day: take(Uint8Array),
    sensor: take(Uint8Array),
    platform: take(Uint8Array),
    product: take(Uint8Array),
    daynight: take(Uint8Array),
    confidence: take(Uint8Array),
    status: take(Uint8Array),
  };
  if (off !== buf.byteLength) throw new Error("month file length does not match its header");
  return { payload: { n, incidents: header.incidents, cols }, transfer };
}

ctx.onmessage = async (e: MessageEvent<Request>) => {
  const { id, kind, url } = e.data;
  try {
    const bytes = await fetchGunzipped(url);
    if (kind === "month") {
      const { payload, transfer } = decodeMonth(bytes);
      ctx.postMessage({ id, data: payload }, transfer);
    } else {
      ctx.postMessage({ id, data: JSON.parse(new TextDecoder().decode(bytes)) });
    }
  } catch (err) {
    ctx.postMessage({ id, error: err instanceof Error ? err.message : String(err) });
  }
};
