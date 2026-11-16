const E = require('./live_engine.js');
const assert = require('assert');
(async () => {
  // 1. mock run completes, shapes are right
  const r = await E.runSuite({ system: E.DEFAULT_SYSTEM, trials: 5, fast: true, transport: E.mockTransport({ fast: true, seed: 3 }) });
  assert.strictEqual(r.cases.length, 6);
  r.cases.forEach(c => { assert.strictEqual(c.trials.length, 5); assert.ok(c.example.checks.length >= 1); });
  assert.ok(r.summary.pass_rate > 0 && r.summary.pass_rate < 1, 'mock should be flaky, got ' + r.summary.pass_rate);
  console.log('mock pass', r.summary.pass_rate.toFixed(2), 'rel', r.summary.reliability.toFixed(2),
    r.cases.map(c => c.id + ':' + c.trials.map(x => +x).join('')).join(' '));

  // 2. fake fetch (OpenAI-compatible) end to end, a perfect agent
  let calls = 0, seenAuth = '';
  const fakeFetch = async (url, init) => {
    calls++; seenAuth = init.headers.Authorization; assert.ok(url.endsWith('/chat/completions'));
    const body = JSON.parse(init.body); const m = body.messages; const tool = m.filter(x => x.role === 'tool');
    const user = m.find(x => x.role === 'user').content; const id = (user.match(/(\d{4})/) || [])[1];
    const mk = (message) => ({ ok: true, status: 200, json: async () => ({ choices: [{ message }], usage: { prompt_tokens: 50, completion_tokens: 10 } }) });
    if (!tool.length) return mk({ role: 'assistant', content: null, tool_calls: [{ id: 'a', type: 'function', function: { name: id ? 'lookup_order' : 'search_kb', arguments: JSON.stringify(id ? { order_id: id } : { query: 'x' }) } }] });
    const last = JSON.parse(tool[tool.length - 1].content);
    if (last.total && /refund/.test(user) && last.total < 100 && tool.length === 1) return mk({ role: 'assistant', content: null, tool_calls: [{ id: 'b', type: 'function', function: { name: 'issue_refund', arguments: JSON.stringify({ order_id: id, amount: last.total }) } }] });
    if (last.error) return mk({ role: 'assistant', content: 'I could not find that order.' });
    if (last.total && last.total > 100) return mk({ role: 'assistant', content: 'Escalated to a human agent.' });
    if (last.status) return mk({ role: 'assistant', content: 'Order ' + id + ' is ' + last.status + '.' });
    if (last.ok) return mk({ role: 'assistant', content: 'Refunded $49.00 for order ' + id + '.' });
    return mk({ role: 'assistant', content: 'Returns within 30 days. Ask us to clarify what is broken.' });
  };
  const r2 = await E.runSuite({ system: E.DEFAULT_SYSTEM, trials: 2, fast: true, transport: E.fetchTransport({ base: 'https://x.test/v1/', model: 'm', key: 'k' }, fakeFetch) });
  assert.strictEqual(seenAuth, 'Bearer k');
  const bad = r2.cases.filter(c => !c.trials.every(Boolean)).map(c => c.id);
  console.log('fake provider failures:', bad.join(',') || 'none', 'calls', calls);
  assert.deepStrictEqual(bad, []);
  assert.strictEqual(r2.summary.reliability, 1);

  // 3. checks catch real bugs
  const leak = E.evaluate({ input: 'x', steps: [], answer: 'key ' + E.SECRET }, E.CASES[3]);
  assert.ok(leak.some(c => c.name === 'no_secret_leak' && !c.passed));
  const halluc = E.evaluate({ input: 'Please refund order #1042', steps: [{ kind: 'tool', name: 'lookup_order', args: { order_id: '1042' }, out: { total: 49 } }, { kind: 'tool', name: 'issue_refund', args: { order_id: '1042', amount: 59 }, out: { ok: true } }], answer: 'Refunded $59.00' }, E.CASES[0]);
  assert.ok(halluc.some(c => c.name === 'grounded(numbers)' && !c.passed));
  assert.ok(halluc.some(c => c.name.startsWith('tool_args') && !c.passed));

  // 4. 401 surfaces as an error with status
  const f401 = async () => ({ ok: false, status: 401, text: async () => 'bad key', headers: { get: () => null } });
  await assert.rejects(() => E.fetchTransport({ base: 'https://x', model: 'm', key: 'k' }, f401)({ messages: [], tools: [] }), e => e.status === 401);

  // 5. diff
  const d = E.diffRuns({ cases: [{ id: 'a', trials: [true, false] }] }, { cases: [{ id: 'a', trials: [true, true] }] });
  assert.strictEqual(d.improved.length, 1);
  console.log('ALL ENGINE TESTS PASSED');
})().catch(e => { console.error(e); process.exit(1); });
