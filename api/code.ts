import { createHash, createHmac, randomBytes, timingSafeEqual } from 'node:crypto'
import type { IncomingMessage, ServerResponse } from 'node:http'
import { getVercelOidcToken } from '@vercel/oidc'

const COOKIE = '__Host-jointinspect-code'
const SESSION_SECONDS = 8 * 60 * 60
const CONTROLLER_URL = (process.env.GPU_CONTROLLER_URL || '').replace(/\/$/, '')
const PROJECT_NUMBER = process.env.GCP_PROJECT_NUMBER || '567370443508'
const POOL_ID = process.env.GCP_WORKLOAD_IDENTITY_POOL_ID || 'vercel'
const PROVIDER_ID = process.env.GCP_WORKLOAD_IDENTITY_POOL_PROVIDER_ID || 'vercel-production'
const SERVICE_ACCOUNT = process.env.GCP_SERVICE_ACCOUNT_EMAIL || 'joint-inspect-vercel-invoker@joint-inspection-510310.iam.gserviceaccount.com'
type Payload = Record<string, unknown>
const GITHUB_REPOSITORY = process.env.GITHUB_CONTROL_REPOSITORY || 'Ifeoyewole/Don-t'

async function callGitHub(path: string, method: 'GET' | 'POST', requestBody?: Payload) {
  const token = process.env.GITHUB_CONTROL_TOKEN
  if (!token) throw new Error('GITHUB_CONTROL_TOKEN is not configured')
  const response = await fetch(`https://api.github.com/repos/${GITHUB_REPOSITORY}${path}`, {
    method,
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: 'application/vnd.github+json',
      'X-GitHub-Api-Version': '2022-11-28',
      ...(requestBody ? { 'Content-Type': 'application/json' } : {}),
    },
    body: requestBody ? JSON.stringify(requestBody) : undefined,
    signal: AbortSignal.timeout(20_000),
  })
  const text = await response.text()
  let payload: Payload
  try { payload = text ? JSON.parse(text) as Payload : {} } catch { payload = { error: 'GitHub returned an invalid response' } }
  return { status: response.status, payload }
}

function send(res: ServerResponse, status: number, body: Payload) {
  const data = JSON.stringify(body)
  res.writeHead(status, { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data), 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' })
  res.end(data)
}

async function body(req: IncomingMessage, maxBytes = 8192): Promise<Payload> {
  const chunks: Buffer[] = []
  let size = 0
  for await (const chunk of req) {
    const value = Buffer.from(chunk)
    size += value.length
    if (size > maxBytes) throw new Error('Request too large')
    chunks.push(value)
  }
  return chunks.length ? JSON.parse(Buffer.concat(chunks).toString('utf8')) as Payload : {}
}

function equal(left: string, right: string) {
  const a = Buffer.from(left)
  const b = Buffer.from(right)
  return a.length === b.length && timingSafeEqual(a, b)
}

function sign(value: string) {
  return createHmac('sha256', process.env.CODE_SESSION_SECRET || '').update(value).digest('base64url')
}

function authenticated(req: IncomingMessage) {
  if (!process.env.CODE_SESSION_SECRET) return false
  const raw = req.headers.cookie?.split(';').map((item) => item.trim()).find((item) => item.startsWith(`${COOKIE}=`))?.slice(COOKIE.length + 1)
  const parts = raw?.split('.') || []
  if (parts.length !== 3) return false
  const value = `${parts[0]}.${parts[1]}`
  return Number(parts[0]) > Date.now() / 1000 && equal(parts[2], sign(value))
}

function sameOrigin(req: IncomingMessage) {
  if (!req.headers.origin || !req.headers.host) return true
  try { return new URL(req.headers.origin).host === req.headers.host } catch { return false }
}

async function callController(path: string, method: 'GET' | 'POST', requestBody?: Payload) {
  if (!CONTROLLER_URL) throw new Error('GPU_CONTROLLER_URL is not configured')
  const oidcToken = await getVercelOidcToken()
  const audience = `//iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_ID}/providers/${PROVIDER_ID}`
  const stsResponse = await fetch('https://sts.googleapis.com/v1/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify({
      grant_type: 'urn:ietf:params:oauth:grant-type:token-exchange',
      audience,
      scope: 'https://www.googleapis.com/auth/cloud-platform',
      requested_token_type: 'urn:ietf:params:oauth:token-type:access_token',
      subject_token_type: 'urn:ietf:params:oauth:token-type:id_token',
      subject_token: oidcToken,
    }),
  })
  if (!stsResponse.ok) throw new Error(`Google identity exchange failed (${stsResponse.status})`)
  const sts = await stsResponse.json() as { access_token: string }
  const tokenResponse = await fetch(`https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/${encodeURIComponent(SERVICE_ACCOUNT)}:generateIdToken`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${sts.access_token}`, 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify({ audience: CONTROLLER_URL, includeEmail: true }),
  })
  if (!tokenResponse.ok) throw new Error(`Cloud Run identity token generation failed (${tokenResponse.status})`)
  const token = ((await tokenResponse.json()) as { token: string }).token
  if (!token) throw new Error('Vercel workload identity is unavailable')
  const response = await fetch(`${CONTROLLER_URL}${path}`, { method, headers: { Authorization: `Bearer ${token}`, Accept: 'application/json', ...(requestBody ? { 'Content-Type': 'application/json' } : {}) }, body: requestBody ? JSON.stringify(requestBody) : undefined, signal: AbortSignal.timeout(25_000) })
  const text = await response.text()
  let payload: Payload
  try { payload = JSON.parse(text) as Payload } catch { payload = { error: 'Controller returned an invalid response' } }
  return { status: response.status, payload }
}

export default async function handler(req: IncomingMessage, res: ServerResponse) {
  res.setHeader('Vary', 'Cookie')
  if (!sameOrigin(req)) return send(res, 403, { error: 'Cross-origin request rejected' })
  const requestUrl = new URL(req.url || '/api/code', `https://${req.headers.host || 'localhost'}`)
  const action = requestUrl.searchParams.get('action') || 'status'

  if (action === 'login' && req.method === 'POST') {
    if (!process.env.CODE_ACCESS_PASSWORD || !process.env.CODE_SESSION_SECRET) return send(res, 503, { error: 'Operator authentication is not configured' })
    try {
      const supplied = String((await body(req)).password || '')
      const actual = createHash('sha256').update(supplied).digest('hex')
      const expected = createHash('sha256').update(process.env.CODE_ACCESS_PASSWORD).digest('hex')
      if (!equal(actual, expected)) return send(res, 401, { error: 'Invalid operator credentials' })
      const value = `${Math.floor(Date.now() / 1000) + SESSION_SECONDS}.${randomBytes(16).toString('base64url')}`
      res.setHeader('Set-Cookie', `${COOKIE}=${value}.${sign(value)}; Path=/; Max-Age=${SESSION_SECONDS}; HttpOnly; Secure; SameSite=Strict`)
      return send(res, 200, { authenticated: true })
    } catch { return send(res, 400, { error: 'Invalid login request' }) }
  }
  if (action === 'logout' && req.method === 'POST') {
    res.setHeader('Set-Cookie', `${COOKIE}=; Path=/; Max-Age=0; HttpOnly; Secure; SameSite=Strict`)
    return send(res, 200, { authenticated: false })
  }
  if (!authenticated(req)) return send(res, 401, { authenticated: false, error: 'Operator authentication required' })

  if (action === 'github_inventory' && req.method === 'GET') {
    try {
      const [pulls, workflows] = await Promise.all([
        callGitHub('/pulls?state=open&per_page=50', 'GET'),
        callGitHub('/actions/workflows?per_page=100', 'GET'),
      ])
      if (pulls.status >= 400) return send(res, pulls.status, pulls.payload)
      if (workflows.status >= 400) return send(res, workflows.status, workflows.payload)
      return send(res, 200, { repository: GITHUB_REPOSITORY, pulls: pulls.payload, workflows: workflows.payload })
    } catch (error) {
      return send(res, 502, { error: error instanceof Error ? error.message : 'GitHub request failed' })
    }
  }
  if (['github_dispatch', 'github_issue'].includes(action) && req.method === 'POST') {
    try {
      const input = await body(req, 64 * 1024)
      if (action === 'github_dispatch') {
        const workflow = String(input.workflow_id || '')
        const ref = String(input.ref || '')
        const expected = `github_dispatch:${workflow}:${ref}`
        if (!workflow || !ref || input.confirmation !== expected) return send(res, 409, { error: `Confirmation must exactly equal ${expected}` })
        const result = await callGitHub(`/actions/workflows/${encodeURIComponent(workflow)}/dispatches`, 'POST', { ref, inputs: input.inputs || {} })
        return send(res, result.status, result.payload)
      }
      const title = String(input.title || '').trim()
      const expected = `github_issue:${title}`
      if (!title || input.confirmation !== expected) return send(res, 409, { error: `Confirmation must exactly equal ${expected}` })
      const result = await callGitHub('/issues', 'POST', { title, body: String(input.body || '') })
      return send(res, result.status, result.payload)
    } catch (error) {
      return send(res, 502, { error: error instanceof Error ? error.message : 'GitHub request failed' })
    }
  }

  const source = requestUrl.searchParams.get('source') || 'build'
  if (!['build', 'batch', 'run'].includes(source)) return send(res, 400, { error: 'Unsupported log source' })
  const prefix = encodeURIComponent(requestUrl.searchParams.get('prefix') || '')
  const service = encodeURIComponent(requestUrl.searchParams.get('service') || '')
  const region = encodeURIComponent(requestUrl.searchParams.get('region') || '')
  const allowed = {
    status: ['/status', 'GET'],
    report: ['/report', 'GET'],
    diagnostics: ['/diagnostics', 'GET'],
    logs: [`/logs?source=${source}&limit=150`, 'GET'],
    inventory: [`/inventory?prefix=${prefix}&service=${service}&region=${region}`, 'GET'],
    run: ['/run', 'POST'],
    upload: ['/upload', 'POST'],
    command: ['/command', 'POST'],
  } as const
  const target = allowed[action as keyof typeof allowed]
  if (!target || req.method !== target[1]) return send(res, 405, { error: 'Operation is not allowed' })
  try {
    const requestBody = ['upload', 'command'].includes(action) ? await body(req, action === 'upload' ? 3 * 1024 * 1024 : 64 * 1024) : undefined
    const result = await callController(target[0], target[1], requestBody)
    return send(res, result.status, result.payload)
  } catch (error) {
    return send(res, 502, { error: error instanceof Error ? error.message : 'Controller request failed' })
  }
}
