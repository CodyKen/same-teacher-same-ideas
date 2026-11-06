"""A self-contained HTML page for one annotator's rewrite-check sheet.

The page shows the rubric and every item (prompt, original, rewrite) with Y/N buttons and a
notes box. Labels are kept in the browser's local storage while the annotator works (a banner
says so when the browser does not allow it for local files), and
"Export CSV" downloads a file in the sheet's CSV format, which `rewrite_check score` reads.
"Import CSV" loads an exported file back, for example in another browser. Item text is placed
with textContent, never parsed as HTML.
"""
import json

LABEL_QUESTIONS = [
    ("kept", "Kept", "Is every claim, example and recommendation of the original in the rewrite?"),
    ("added", "Added", "Does the rewrite state a claim or example that is not in the original?"),
    ("plain", "Plain", "Is the rewrite plain prose, with no lists, headings, bold text, emoji or marked voice?"),
]

RUBRIC_HTML = """
<p>Each item shows a prompt, an <b>original</b> answer and a <b>rewrite</b>. The rewrite was
asked to restate the original in plain prose and keep every claim and example. Judge the
content, not the quality of the writing. Answer three questions with Y or N, and use the notes
box for anything unclear.</p>
<dl>
  <dt>Kept</dt>
  <dd><b>Y</b> if every claim, example and recommendation of the original appears in the rewrite.
  Rewording, merging sentences and dropping formatting are fine. <b>N</b> if anything of
  substance is missing or its meaning changed.</dd>
  <dt>Added</dt>
  <dd><b>Y</b> if the rewrite states a claim or example that is not in the original. Connecting
  words and restated context from the prompt do not count. Otherwise <b>N</b>.</dd>
  <dt>Plain</dt>
  <dd><b>Y</b> if the rewrite is plain prose: no lists, headings, bold text or emoji, and no
  marked voice such as jokes, exclamations or a persona. Otherwise <b>N</b>.</dd>
</dl>
<p>Work alone, and do not discuss items until every sheet is in. Do not open
<code>key_DO_NOT_OPEN.json</code>.</p>
<p>Labels are saved in this browser as you go. When you finish, click <b>Export CSV</b> and put
the downloaded file in <code>runs/human_eval/rewrite_fidelity/</code>, replacing the blank sheet
of the same name. Export also works as a backup at any time; <b>Import CSV</b> loads it back.</p>
"""

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rewrite Check __SHEET_TITLE__</title>
<style>
:root {
  --bg: #f6f6f3; --card: #ffffff; --fg: #1d1d1b; --muted: #6b6b66; --border: #dcdcd6;
  --accent: #2f5d8a; --accent-soft: #e6eef6; --yes: #2e7d4f; --no: #b3412f; --done: #eaf4ee;
  --prompt-bg: #f1efe7;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #161615; --card: #1f1f1d; --fg: #ecece8; --muted: #a2a29b; --border: #3a3a36;
    --accent: #8db7e0; --accent-soft: #24313d; --yes: #6fcf97; --no: #f08b78; --done: #1e2a22;
    --prompt-bg: #262520;
  }
}
:root[data-theme="dark"] {
  --bg: #161615; --card: #1f1f1d; --fg: #ecece8; --muted: #a2a29b; --border: #3a3a36;
  --accent: #8db7e0; --accent-soft: #24313d; --yes: #6fcf97; --no: #f08b78; --done: #1e2a22;
  --prompt-bg: #262520;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg);
  font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "PingFang SC", "Microsoft YaHei", sans-serif; }
header { position: sticky; top: 0; z-index: 5; background: var(--card); border-bottom: 1px solid var(--border); }
.bar { max-width: 1180px; margin: 0 auto; padding: 10px 16px; display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
.bar h1 { font-size: 17px; margin: 0 12px 0 0; }
.progress { flex: 1 1 160px; min-width: 140px; }
.progress .track { height: 8px; background: var(--border); border-radius: 4px; overflow: hidden; }
.progress .fill { height: 100%; width: 0; background: var(--yes); transition: width .2s; }
.progress .text { font-size: 13px; color: var(--muted); margin-top: 2px; }
button, label.button { font: inherit; font-size: 14px; padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border);
  background: var(--card); color: var(--fg); cursor: pointer; }
button.primary { background: var(--accent); border-color: var(--accent); color: var(--card); }
button:hover, label.button:hover { border-color: var(--accent); }
main { max-width: 1180px; margin: 0 auto; padding: 16px; }
details.rubric { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 12px 16px; margin-bottom: 18px; }
details.rubric summary { cursor: pointer; font-weight: 600; }
details.rubric dt { font-weight: 600; margin-top: 8px; }
details.rubric dd { margin: 2px 0 0 0; }
code { font-size: 0.9em; background: var(--accent-soft); padding: 1px 4px; border-radius: 4px; }
.item { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; margin-bottom: 18px; scroll-margin-top: 120px; }
.item.complete { background: var(--done); }
.item .head { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; margin-bottom: 8px; }
.item .id { font-weight: 700; }
.item .status { font-size: 13px; color: var(--muted); }
.prompt { background: var(--prompt-bg); border-radius: 8px; padding: 10px 12px; margin-bottom: 12px; white-space: pre-wrap; overflow-wrap: anywhere; }
.prompt .tag, .text .tag { display: block; font-size: 12px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; color: var(--muted); margin-bottom: 4px; }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
@media (max-width: 760px) { .pair { grid-template-columns: 1fr; } }
.text { border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; min-width: 0; }
.text .body { white-space: pre-wrap; overflow-wrap: anywhere; }
.text .count { font-weight: 400; text-transform: none; letter-spacing: 0; }
.labels { margin-top: 12px; display: grid; gap: 8px; }
.q { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; }
.q .name { font-weight: 700; width: 56px; }
.q .question { flex: 1 1 280px; color: var(--muted); font-size: 14px; }
.choices { display: inline-flex; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
.choices input { position: absolute; opacity: 0; pointer-events: none; }
.choices label { padding: 4px 16px; cursor: pointer; font-weight: 600; user-select: none; }
.choices label + input + label { border-left: 1px solid var(--border); }
.choices input:focus-visible + label { outline: 2px solid var(--accent); outline-offset: -2px; }
.choices input.y:checked + label { background: var(--yes); color: var(--card); }
.choices input.n:checked + label { background: var(--no); color: var(--card); }
textarea { width: 100%; min-height: 38px; resize: vertical; font: inherit; font-size: 14px; padding: 6px 8px;
  border: 1px solid var(--border); border-radius: 6px; background: var(--card); color: var(--fg); }
.hidden { display: none; }
.warning { background: #fff4d6; color: #5a4300; border: 1px solid #e6c766; border-radius: 8px; padding: 10px 14px; margin-bottom: 16px; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) .warning { background: #3a3014; color: #f3dc9a; border-color: #7a6424; } }
:root[data-theme="dark"] .warning { background: #3a3014; color: #f3dc9a; border-color: #7a6424; }
.toast { position: fixed; bottom: 16px; left: 50%; transform: translateX(-50%); background: var(--fg); color: var(--bg);
  padding: 8px 14px; border-radius: 6px; font-size: 14px; opacity: 0; transition: opacity .2s; pointer-events: none; }
.toast.show { opacity: 1; }
</style>
</head>
<body>
<header>
  <div class="bar">
    <h1>Rewrite check, __SHEET_TITLE__</h1>
    <div class="progress"><div class="track"><div class="fill" id="fill"></div></div><div class="text" id="progress-text"></div></div>
    <button id="next">Next unfinished</button>
    <button id="filter" aria-pressed="false">Show unfinished only</button>
    <label class="button" for="import">Import CSV</label>
    <input type="file" id="import" accept=".csv,text/csv" class="hidden">
    <button class="primary" id="export">Export CSV</button>
  </div>
</header>
<main>
  <div class="warning hidden" id="storage-warning">This browser does not save progress for this page. Labels will be lost if you close or reload it, so click <b>Export CSV</b> often, or open the page in Chrome or Safari.</div>
  <details class="rubric" id="rubric" open>
    <summary>Instructions</summary>
    __RUBRIC__
  </details>
  <div id="items"></div>
</main>
<div class="toast" id="toast"></div>
<script type="application/json" id="data">__DATA__</script>
<script>
(function () {
  "use strict";
  var DATA = JSON.parse(document.getElementById("data").textContent);
  var LABELS = DATA.labels;
  var STORE = "rewrite-check:" + DATA.sheet;
  var state = {}, storage = true;
  try {
    localStorage.setItem(STORE + ":probe", "1"); localStorage.removeItem(STORE + ":probe");
    state = JSON.parse(localStorage.getItem(STORE) || "{}") || {};
  } catch (e) { state = {}; storage = false; }
  if (!storage) document.getElementById("storage-warning").classList.remove("hidden");
  var onlyUnfinished = false;

  function save() {
    if (storage) { try { localStorage.setItem(STORE, JSON.stringify(state)); } catch (e) { storage = false; document.getElementById("storage-warning").classList.remove("hidden"); } }
    refresh();
  }
  function entry(id) { if (!state[id]) state[id] = {}; return state[id]; }
  function complete(id) { var s = state[id] || {}; return LABELS.every(function (l) { return s[l[0]] === "Y" || s[l[0]] === "N"; }); }
  function words(t) { var m = t.trim().match(/\S+/g); return m ? m.length : 0; }
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }

  function toast(msg) {
    var t = document.getElementById("toast"); t.textContent = msg; t.classList.add("show");
    clearTimeout(toast.timer); toast.timer = setTimeout(function () { t.classList.remove("show"); }, 2200);
  }

  function textBox(tag, body) {
    var box = el("div", "text");
    var head = el("span", "tag", tag + " ");
    head.appendChild(el("span", "count", "(" + words(body) + " words)"));
    box.appendChild(head);
    box.appendChild(el("div", "body", body));
    return box;
  }

  function render() {
    var root = document.getElementById("items");
    root.textContent = "";
    DATA.items.forEach(function (item, n) {
      var card = el("section", "item"); card.id = "item-" + item.item_id; card.dataset.id = item.item_id;
      var head = el("div", "head");
      head.appendChild(el("span", "id", (n + 1) + " / " + DATA.items.length + "  ·  " + item.item_id));
      head.appendChild(el("span", "status"));
      card.appendChild(head);
      var prompt = el("div", "prompt"); prompt.appendChild(el("span", "tag", "Prompt")); prompt.appendChild(document.createTextNode(item.prompt || "(no prompt text)"));
      card.appendChild(prompt);
      var pair = el("div", "pair");
      pair.appendChild(textBox("Original", item.original));
      pair.appendChild(textBox("Rewrite", item.rewrite));
      card.appendChild(pair);
      var labels = el("div", "labels");
      LABELS.forEach(function (l) {
        var key = l[0], q = el("div", "q");
        q.appendChild(el("span", "name", l[1]));
        var choices = el("span", "choices");
        ["Y", "N"].forEach(function (v) {
          var id = item.item_id + "-" + key + "-" + v;
          var input = document.createElement("input");
          input.type = "radio"; input.name = item.item_id + "-" + key; input.id = id; input.value = v; input.className = v.toLowerCase();
          input.checked = (state[item.item_id] || {})[key] === v;
          input.addEventListener("change", function () { entry(item.item_id)[key] = v; save(); });
          var label = el("label", null, v); label.htmlFor = id;
          choices.appendChild(input); choices.appendChild(label);
        });
        q.appendChild(choices);
        q.appendChild(el("span", "question", l[2]));
        labels.appendChild(q);
      });
      var notes = document.createElement("textarea");
      notes.placeholder = "Notes (optional)"; notes.value = (state[item.item_id] || {}).notes || "";
      notes.addEventListener("input", function () { entry(item.item_id).notes = notes.value; save(); });
      labels.appendChild(notes);
      card.appendChild(labels);
      root.appendChild(card);
    });
    refresh();
  }

  function refresh() {
    var done = 0;
    DATA.items.forEach(function (item) {
      var ok = complete(item.item_id); if (ok) done++;
      var card = document.getElementById("item-" + item.item_id); if (!card) return;
      card.classList.toggle("complete", ok);
      card.querySelector(".status").textContent = ok ? "done" : "to do";
      card.classList.toggle("hidden", onlyUnfinished && ok);
    });
    document.getElementById("fill").style.width = (100 * done / DATA.items.length) + "%";
    document.getElementById("progress-text").textContent = done + " of " + DATA.items.length + " items labeled";
  }

  function csvCell(v) { v = v == null ? "" : String(v); return /[",\r\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }
  function exportCsv() {
    var header = ["item_id", "prompt", "original", "rewrite"].concat(LABELS.map(function (l) { return l[0]; }), ["notes"]);
    var lines = [header.join(",")];
    DATA.items.forEach(function (item) {
      var s = state[item.item_id] || {};
      var row = [item.item_id, item.prompt, item.original, item.rewrite].concat(LABELS.map(function (l) { return s[l[0]] || ""; }), [s.notes || ""]);
      lines.push(row.map(csvCell).join(","));
    });
    var blob = new Blob([lines.join("\r\n") + "\r\n"], { type: "text/csv;charset=utf-8" });
    var a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "sheet_" + DATA.sheet + ".csv";
    document.body.appendChild(a); a.click(); a.remove(); setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
    var left = DATA.items.length - DATA.items.filter(function (i) { return complete(i.item_id); }).length;
    toast(left ? "Exported. " + left + " items still unlabeled." : "Exported. All items labeled.");
  }

  function parseCsv(text) {
    var rows = [], row = [], cell = "", i = 0, quoted = false;
    if (text.charCodeAt(0) === 0xFEFF) i = 1;
    for (; i < text.length; i++) {
      var c = text[i];
      if (quoted) {
        if (c === '"') { if (text[i + 1] === '"') { cell += '"'; i++; } else quoted = false; } else cell += c;
      } else if (c === '"') quoted = true;
      else if (c === ",") { row.push(cell); cell = ""; }
      else if (c === "\n" || c === "\r") { if (c === "\r" && text[i + 1] === "\n") i++; row.push(cell); rows.push(row); row = []; cell = ""; }
      else cell += c;
    }
    if (cell !== "" || row.length) { row.push(cell); rows.push(row); }
    return rows;
  }
  function importCsv(file) {
    var reader = new FileReader();
    reader.onload = function () {
      var rows = parseCsv(String(reader.result)); if (!rows.length) return;
      var head = rows[0], col = {}; head.forEach(function (h, j) { col[h.trim()] = j; });
      var known = {}; DATA.items.forEach(function (i) { known[i.item_id] = true; });
      var n = 0;
      rows.slice(1).forEach(function (r) {
        var id = r[col.item_id]; if (!known[id]) return;
        var e = entry(id);
        LABELS.forEach(function (l) { var v = (r[col[l[0]]] || "").trim().toUpperCase().charAt(0); if (v === "Y" || v === "N") e[l[0]] = v; });
        if (col.notes != null && r[col.notes]) e.notes = r[col.notes];
        n++;
      });
      save(); render(); toast("Imported labels for " + n + " items.");
    };
    reader.readAsText(file, "utf-8");
  }

  document.getElementById("export").addEventListener("click", exportCsv);
  document.getElementById("import").addEventListener("change", function (e) { if (e.target.files[0]) importCsv(e.target.files[0]); e.target.value = ""; });
  document.getElementById("next").addEventListener("click", function () {
    var next = DATA.items.find(function (i) { return !complete(i.item_id); });
    if (next) document.getElementById("item-" + next.item_id).scrollIntoView({ behavior: "smooth", block: "start" });
    else toast("Every item is labeled.");
  });
  document.getElementById("filter").addEventListener("click", function (e) {
    onlyUnfinished = !onlyUnfinished; e.target.setAttribute("aria-pressed", String(onlyUnfinished));
    e.target.textContent = onlyUnfinished ? "Show all items" : "Show unfinished only"; refresh();
  });
  render();
})();
</script>
</body>
</html>
"""


def render(sheet: str, items: list[dict]) -> str:
    """HTML for one sheet. `sheet` is the file stem suffix (a1, a2, ...); items come from its CSV."""
    payload = {
        "sheet": sheet,
        "labels": LABEL_QUESTIONS,
        "items": [{k: item[k] for k in ("item_id", "prompt", "original", "rewrite")} for item in items],
    }
    # Escape "<" so no answer text can close the script element.
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    title = f"Annotator {sheet[1:]}" if sheet.startswith("a") else sheet
    return TEMPLATE.replace("__SHEET_TITLE__", title).replace("__RUBRIC__", RUBRIC_HTML).replace("__DATA__", data)
