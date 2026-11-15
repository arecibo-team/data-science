// Fetches /api/points and decodes it off the main thread.
// Columns are decoded straight into typed arrays, which are transferred (not copied) back.
// The browser already gunzips the response (Content-Encoding), so the body is plain bytes.

import type { Columns, PointsData } from "./types";

interface Request {
  id: number;
  url: string;
}

// Untyped view of the worker global, so this file does not clash with the DOM lib types.
const ctx = self as unknown as {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  onmessage: ((e: MessageEvent<Request>) => void) | null;
};

/**
 * Decode a points response written by server/app.py: a uint32 header length, a JSON header,
 * then each column as raw little-endian bytes in a fixed order.
 */
function decode(buf: ArrayBuffer) {
  const hlen = new DataView(buf).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hlen))) as {
    n: number;
    total: number;
    overflow: boolean;
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
  // Must match COLUMNS in server/app.py.
  const cols: Columns = {
    lat: take(Float32Array),
    lon: take(Float32Array),
    frp: take(Float32Array),
    bt: take(Float32Array),
    cluster: take(Int32Array),
    ts: take(Uint32Array),
    incident: take(Uint16Array),
    sensor: take(Uint8Array),
    platform: take(Uint8Array),
    product: take(Uint8Array),
    daynight: take(Uint8Array),
    confidence: take(Uint8Array),
    status: take(Uint8Array),
  };
  if (off !== buf.byteLength) throw new Error("points response length does not match its header");
  const payload: PointsData = { n, total: header.total, overflow: header.overflow, incidents: header.incidents, cols };
  return { payload, transfer };
}

ctx.onmessage = async (e: MessageEvent<Request>) => {
  const { id, url } = e.data;
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${new URL(url).pathname}: HTTP ${res.status}`);
    const { payload, transfer } = decode(await res.arrayBuffer());
    ctx.postMessage({ id, data: payload }, transfer);
  } catch (err) {
    ctx.postMessage({ id, error: err instanceof Error ? err.message : String(err) });
  }
};
