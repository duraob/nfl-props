# How This Project Works

This guide explains how to run the project and, in plain language, how it actually
works and why we trust it. It's written so someone with no programming or stats
background can follow along.

---

## Section 1: Commands and Schedule

Always start by turning on the project's Python environment. Every command below
assumes you've done this first, in the project folder:

```bash
source .venv/bin/activate
```

### There is no fixed weekly schedule — an NFL week is not one event

An early version of this guide had you capturing prices "Wednesday and Sunday."
That's wrong, and worth explaining why: a real NFL week can have games on
Wednesday, Thursday, Friday (occasionally), Sunday, *and* Monday. If you only check
prices twice on a fixed schedule, you'll completely miss the right moment for
whichever games don't fall on those two days — by the time "Sunday morning"
arrives, Thursday's game is already over and gone for good, with no way to
recapture what the closing price was.

So instead, ask the system what the actual game days are for the week you care
about:

```python
import projections as P
P.kickoff_windows(2026, 1)
```

This prints one row per distinct day games are happening that week, with a
suggested time to take action for each — 3 hours before the earliest kickoff that
day, giving you room to actually place something before lines move further. For a
real Week 1 that might look like:

| Day | Games | Earliest kickoff | Take action by |
|---|---|---|---|
| Wed 9/9 | 1 | 8:20 PM | 5:20 PM |
| Thu 9/10 | 1 | 8:35 PM | 5:35 PM |
| Sun 9/13 | 13 | 1:00 PM | 10:00 AM |
| Mon 9/14 | 1 | 8:15 PM | 5:15 PM |

Run your capture and decision-making once per row, at that row's suggested time —
not on a blanket "twice a week" schedule. Concretely, for each day:

```bash
python odds_capture.py       # Kalshi snapshot - free, run it every time
python dk_capture.py 6       # DraftKings prices for anything kicking off soon
```

**A caveat on the exact times:** the game times are treated as Eastern time
(matching a standard 1:00 PM ET Sunday kickoff, which is what the data shows) — this
matches CT directly, but it's worth double-checking against the actual NFL schedule
once during the season rather than trusting it blindly forever.

### Getting a projection for a player or a week

There's no single button that does everything yet. You ask for numbers in Python:

```python
import projections as P

proj = P.project(season=2026, week=1)
proj.head(20)   # the 20 players expected to be most involved this week
```

This gives you a table with one row per player, and columns for each stat we
predict: expected passing yards, rushing yards, receiving yards, catches,
touchdowns by type, and interceptions — plus two **confidence** columns, explained
in Section 2. There's no single combined "score" — see Section 2 for why we
deliberately don't produce one.

### Getting a short, filtered report instead of a 300-row table

You said you don't have the ammo to act on every bet — this is the tool for that.
It filters the full projection table down to only the players we're genuinely
confident in (see "How confident are we?" in Section 2), grouped by position so
you get a mix of quarterbacks, running backs, receivers, and tight ends instead of
one position crowding out the rest:

```bash
python -c "import report; print(report.build_report(2026, 1, game_date='2026-09-10'))"
```

Leave off `game_date` to get the whole week at once; pass one of the dates from
`kickoff_windows()` above to get just that day's games. To have it sent straight to
your phone instead of printed to a terminal:

```bash
python -c "import report, telegram_notify as T; T.send_message(report.build_report(2026, 1, game_date='2026-09-10'))"
```

This is **still not a finished bet recommendation**, but it's more than a trust
score now: wherever we've actually captured a Kalshi or DraftKings line for a shown
player, the report also prints an **edge** — our estimate of whether the number
beats that price, shown right next to it, e.g. `rec yds 90.7 (edge +9% vs kalshi
80.5) [high]`. Most players won't show one yet, simply because most props haven't
been captured for the week you're asking about — that's expected, not broken. When
neither venue has a captured line for a player, you're back to comparing the
projection against whatever price you see yourself. See "Confidence vs. edge" in
Section 2 for exactly what edge does and doesn't mean, and why it's shown next to
confidence instead of folded into one number.

### Keeping a record of what you predicted and bet

Before you act on a report, log it — this is what lets you check later whether the
model is actually any good, instead of just trusting your memory of it:

```python
import projections as P
import ledger as L

proj = P.project(2026, 1)
L.log_predictions(proj)
```

Do this right before you send yourself the report, not on some fixed schedule — the
timestamp is the point. It's what makes it impossible to quietly fudge a projection
after the fact once you know how the game went.

When you actually place a bet, record it by hand:

```python
import ledger as L

L.record_bet(season=2026, week=1, player_id="00-0033280", stat="rec_yd",
             venue="draftkings", side="over", line=70.5, price=-115, stake=20)
```

`player_id` is the nflverse ID (visible as `player_id` in any projection table).
`side` is always `"over"` or `"under"` — for a Kalshi "yes" bet on a threshold,
record it as `"over"`; for "no", record `"under"`.

Once a week's games are done (Tuesday is a reasonable time), grade everything you
bet that week:

```bash
python settle.py 2026 1
```

This tells you, for each bet: what actually happened, how far off the projection
was, whether the bet won or lost, and — for DraftKings bets on receptions
specifically — whether you beat the closing price (see "Why we record prices twice
per game day" in Section 3). Kalshi bets and DraftKings yardage bets don't get that
last number yet: there's no reliable way to match a manually-recorded bet back to
Kalshi's captured lines (they're stored as plain-English titles, not structured
data), and DraftKings closing lines are only captured for receptions today.

### Checking that the model is still accurate

```bash
python backtest.py
```

This runs the model against past seasons where we already know what happened, and
prints a report card. See Section 2 for how to read it.

### Before you spend real money on data

`dk_capture.py` uses a service (The Odds API) that only gives us 500 free lookups a
month. Always check the cost first:

```bash
python dk_capture.py 6 --dry-run
```

This tells you what it *would* cost without actually spending anything.

### Running this without you having to remember any of it

Everything above is something *you* run by hand, at roughly the right time. That's
fine for trying the system out, but a real season needs captures at specific
kickoff-adjacent moments several days a week — Wednesday evening, Thursday evening,
Sunday morning, Monday evening — and a laptop that's asleep or closed at the wrong
moment misses one of those permanently (betting lines can't be reconstructed after
the fact).

`schedule_captures.py` automates this: it checks whether *right now* falls inside
one of the capture windows `kickoff_windows()` says matter, and if so, captures both
venues, logs the week's predictions, and pushes the report — all in one pass. Run
alone it does nothing most of the time by design:

```bash
python schedule_captures.py
```

The intended setup is a small always-on machine (a $6/month DigitalOcean droplet is
plenty — this workload is light) running that command every 15 minutes via cron, so
it's never more than 15 minutes late to a window even if nothing else is watching
the clock. `deploy/setup.sh` does the one-time setup (installs the right Python
version, creates the virtual environment, sets the machine's clock to Eastern since
that's what kickoff times are recorded in, and installs the cron schedule) — see
the comments at the top of that file for the exact steps. It does **not** create
`.env` for you; secrets get copied over by hand after setup, not generated or
fetched automatically.

The same setup also schedules a weekly report card (`scorecard.py --push`, Tuesday
mornings) — how accurate last week's predictions actually were, plus closing-line
value across every bet recorded so far. This is the thing that eventually answers
"is this actually working," which nothing before Week 1 can tell you.

---

## Section 2: The Projection Engine — What It Does and Why It Works

### The basic idea

Every week, we want to predict specific things about how NFL players will perform
— how many receiving yards a wide receiver will get, how many times a running
back will carry the ball, how many touchdowns a quarterback will throw. We
deliberately predict these numbers directly, rather than combining them into one
overall "score" — different stats are useful for different purposes (fantasy
football, a specific prop bet, checking a game's whole outlook), and combining
them into a single number would throw away information we might need. The
obvious approach to predicting any one of these numbers is: "look at how well a
player's done recently, and guess they'll do about the same." Our system does
that too — but it does one thing smarter than a simple average, and that one
thing is the whole reason it works.

### The key idea: some stats are predictable, some are luck

Imagine two facts about a wide receiver:

1. **How often his team throws him the ball** (his "target share")
2. **How many yards he gets, on average, per catch**

You'd think both are useful for predicting next week. But we tested this on three
years of real NFL data, and the answer surprised us:

- **How often he gets the ball is very consistent, week to week.** If a receiver
  got 25% of his team's throws in the first half of the season, he'll almost
  certainly get around 25% in the second half. This is measurable and stable.
- **How many yards he gets per catch bounces around a lot.** A receiver who
  averaged 15 yards per catch one month might average 9 yards per catch the next
  month, for no real reason — that's just how football works. A lot of it is luck:
  a bounce that turns a 5-yard catch into a 40-yard catch, a missed tackle, etc.
- **Touchdowns are almost entirely luck.** A player's touchdown rate from the
  first half of a season barely predicts his touchdown rate in the second half.
  Touchdowns depend on which specific 5 yards of the field a play happens to end
  on — that's mostly random, even for good players.

We measured exactly *how* predictable each of these is, using a statistics
technique called **split-half reliability**: split each player's games into two
random halves, and see how well one half predicts the other. A score of 1.0 means
"perfectly predictable." A score of 0.0 means "pure luck, no pattern at all."

Here's what we found:

| Stat | How predictable (0 = luck, 1 = perfectly consistent) |
|---|---|
| How often a player gets the ball (targets, carries) | **0.95** — very consistent |
| Yards per catch or per carry | **0.40** — mostly luck, some skill |
| Touchdown rate | **0.09** — almost entirely luck |

### What we do with that

This is the core trick, and it's a well-known statistics idea called **shrinkage**:
when a number is unreliable, don't fully trust it — pull it back toward the
"normal" average for that position. The less reliable the number, the harder you
pull it back.

- **How often a player gets the ball:** we trust the player's own recent numbers
  almost completely (about 95%), because we measured that this is genuinely
  consistent.
- **Yards per catch/carry:** we only trust the player's own recent numbers about
  40% of the time. The other 60% of our guess comes from "what's normal for this
  position." A receiver who's had a hot streak of long catches gets pulled back
  toward a more typical number, because we know that hot streak probably won't
  continue.
- **Touchdowns:** we barely trust the player's own recent touchdown count at all
  (about 9%). Instead we mostly use the *typical* touchdown rate for a player who
  gets that many chances, and we adjust it up or down slightly using the Vegas
  betting line for how many total points that player's team is expected to score
  that week (a high-scoring game means more touchdown chances for everyone).

**Why does this matter?** If you don't do this, a player who got lucky on a couple
of long touchdowns looks like a superstar in the numbers, and a system that just
copies recent performance will predict he keeps doing that — and be wrong, a lot.
By figuring out which parts of a player's stat line are real skill and which parts
are luck, we avoid chasing hot streaks that are about to end.

### How confident are we? (and what that does and doesn't mean)

You asked for a way to tell which projections are worth acting on, since you don't
have the ammo to bet on everything. Here's how that's built, directly from the same
reliability numbers above.

Every projection comes with a confidence score built from two things:

1. **How predictable that category of stat is** (the 0.95 / 0.40 / 0.09 numbers
   from the table above — this sets a hard ceiling).
2. **How many games of history we actually have for that player.** Two games of
   data is a much shakier guess than a full season's worth, even for a highly
   predictable stat like target share.

The confidence score ramps up as more games of history pile up, but it can never
climb higher than the ceiling for that category. Practically, this means:

- **Yardage and catch-count confidence** can reach as high as **0.40** once a
  player has around 8 games of recent history — matching the "yards per catch is
  40% predictable" finding above.
- **Touchdown and interception confidence can never exceed about 0.09**, no
  matter how many games of history exist, because that's the hard ceiling we
  measured for how predictable touchdowns actually are. We label anything below
  about 0.12 as "low" confidence — which means **every touchdown and interception
  prediction is always labeled "low."** That's not a bug. It's the honest
  reflection of "touchdowns are mostly luck" turned into something you can filter
  on. By default, the filtered report described in Section 1 only shows you
  "high" confidence *yardage* numbers — touchdown props are deliberately left out
  of it.

**Confidence vs. edge — an important distinction.** A "high confidence" label
means we trust that *specific number* — it does not mean the bet is a good one. To
know if a bet is actually worth making, you compare our number against the price a
book or Kalshi is offering, and see if there's a gap. That comparison — called
**edge** — is now automatic wherever we've captured a real line: the report in
Section 1 prints it right next to the projection it applies to, e.g. `edge +9% vs
kalshi 80.5`, meaning we think the real probability is about 9 percentage points
higher than the market's price implies. It never replaces confidence or gets
blended into it — they answer different questions. A number can be high-confidence
with no edge at all (the market already prices it correctly); a number can show an
edge while still being low-confidence (which is a *weaker* signal, not a stronger
one — a noisy guess that happens to look off-market isn't the same as a trustworthy
one that is). Edge only shows up where a line was actually captured, which today is
most of the time *not* the case — Kalshi's weekly player-prop markets don't open
until close to each week's games (see Section 3). Where no line has been captured,
treat the confidence filter as narrowing 300 players down to the handful worth
personally checking against a real price yourself — not as a finished
recommendation.

**One more honest caveat, specific to the first two weeks of a season.** Early in
a season, there isn't yet enough *this-season* data to go on, so the system
borrows from the end of last season to fill the gap — otherwise it would have
nothing to say at all for the season's first couple of weeks, which is obviously
not useful. We checked how well this works: it still beats a plain average on 7
of the 8 stats we track, but predictions in weeks 1-2 run a bit higher than
reality on average (more so than later weeks), and interception predictions
specifically are actually a bit worse than just guessing the naive average.
Treat early-season confidence labels as slightly more optimistic than they
technically ought to be — a real, known limitation, not something the numbers
already correct for. It also has no way of knowing when a player has switched
teams over the offseason and inherited a completely different role; it can only
carry forward what he did on his old team.

### Things we deliberately do NOT do

We tested several ideas that sound reasonable but turned out not to help, so we
left them out on purpose:

- **Adjusting for how good the opponent's defense is.** This sounds obviously
  useful, but when we tested it against real outcomes, it made predictions
  *slightly worse*, not better. NFL defenses are inconsistent enough week to week
  that a simple "this defense is weak, expect a big game" adjustment adds more
  noise than signal.
- **Weighting recent games much more heavily than older ones.** We tested many
  different levels of "how much should last week matter more than five weeks ago"
  and found it barely mattered — a small amount of recency weighting is about as
  good as a lot of it.
- **Trying to predict how many plays a team will run based on the Vegas point
  total.** We assumed a team expected to score a lot of points would also run more
  plays. It turns out that's not true — NFL teams run a fairly similar number of
  plays regardless of the score. Vegas point totals *do* predict how many
  touchdowns happen, just not how many total plays happen.
- **Combining everything into one overall "fantasy score."** An earlier version of
  this system did this, matching one specific fantasy platform's scoring rules
  (including some tricky bonus-point math). We removed it — it added real
  complexity for a feature that wasn't actually needed. Predicting each stat on
  its own is simpler, and it's also more directly useful: a specific stat number
  is what a prop bet is priced on, and fantasy scoring can always be calculated
  from the individual stats later, by hand, for whatever league's rules apply.

### How we know it actually works — the report card

The honest way to test a prediction system is to hide some real data from it,
build the system using only the rest, and then check its predictions against the
data it never saw. This is what `backtest.py` does:

- We built the whole system using only 2023 and 2024 data.
- We then tested it against 2025 — a season the system had never seen and that
  had no influence on how it was built.
- If it does about as well on the unseen 2025 season as it did on the seasons it
  was built from, that tells us it has found a real, repeatable pattern — not
  just a coincidence that happened to fit the specific games we built it on.

Results: on the 2025 season it had never seen, the system beat a simple "average
of the last 4 games" prediction on **every single stat we checked, across the
whole season** — passing yards, rushing yards, receiving yards, catches, and
touchdowns of every type. The size of the improvement varied by stat, roughly 3%
to 21% better than the simple average depending which stat you look at, which
lines up with what Section 2 told you to expect: touchdowns are the "mostly luck"
category, so the system's edge there, while real, is a smaller improvement in an
absolute sense than its edge on something more predictable like receiving yards.
(The "How confident are we?" section above has the specific caveat for weeks 1-2,
where one stat — interceptions — actually loses to the simple average, even though
the season as a whole doesn't.)

One honest weak spot: quarterback stats are noticeably harder to predict than
running back, wide receiver, or tight end stats. Quarterback performance depends
heavily on touchdowns and scrambling for extra yards, both of which are the
"mostly luck" categories described above — so there's a real ceiling on how well
any system can predict quarterbacks. And touchdowns and interceptions specifically
are hard to predict well for *any* position — the system still beats a plain
average on them, but neither it nor any other approach gets them very accurate in
an absolute sense, because so much of a touchdown or an interception comes down to
one random moment in a game.

---

## Section 3: Where the Data Comes From

### Player stats and game data — nflverse

**What it is:** A free, publicly maintained collection of NFL statistics, built by
volunteers and widely used across the football analytics community.

**What we get from it:** Every player's stats for every game — passing, rushing,
receiving, plus more detailed numbers like how many snaps a player was on the
field for, and what percentage of his team's targets or carries went to him.

**Why we trust it:** We tested it directly rather than just taking someone's word
for it. Before switching to this source, our previous system scraped data directly
from websites, and had a serious hidden problem: for over half of all player-games,
the "percentage of snaps played" number was missing or broken, because website
scraping is fragile and things quietly break. That flaw once caused our system to
accidentally throw out 80% of NFL players from its calculations, including several
of the best wide receivers in the league, without any error message — it just
silently produced bad results. We checked nflverse's version of the same
information and found it was missing for less than 6% of players, a huge
improvement. We also confirmed the specific players who went missing before
(A.J. Brown, Ja'Marr Chase, Justin Jefferson, and others) are correctly included
now.

**How often it updates:** During the season, this data typically updates within
hours after games finish. Our system checks a timestamp this data source publishes
about itself, and will refuse to run if the data looks too old — so we don't
accidentally build predictions on stale information without knowing it.

### Betting lines — Kalshi

**What it is:** A regulated exchange (like a stock market, but for predictions)
where people trade contracts on real-world outcomes, including NFL player and game
results.

**What we get from it:** Prices that represent the market's best guess at the
probability of something happening — for example, "will this receiver get 50+
receiving yards this week?" Because it's a real exchange with buyers and sellers
setting prices, and because it charges very low fees (roughly 1%, versus roughly
4-5% at a typical sportsbook), a Kalshi price tends to be a more honest estimate of
the true probability than a sportsbook's number, which has more built-in margin
baked in.

**Why we trust it:** It requires no paid account or subscription to check prices —
public information, and we can verify it's genuinely live by checking it against
real games as they happen. Earlier testing also compared our model's predictions
against Kalshi's exact price levels and found them to line up closely — a good
sign that the model and this market broadly agree on what's likely. Our system
predicts individual stats directly (like "142 receiving yards"), and separately
converts that into a probability of clearing whatever specific line Kalshi is
actually offering (like "will this player get 50+ yards?") — that conversion, and
the comparison against a captured Kalshi price, is what shows up as **edge** in the
report described in Section 1.

**What we've confirmed it offers:** Kalshi does run weekly, single-game bets on
receiving yards, rushing yards, and passing yards — we confirmed this by finding
real, already-settled bets from this past week's preseason games (for example,
"will this player get 50+ receiving yards in this specific game?"). These weekly
markets sit empty between game weeks and reopen closer to the next slate of games,
which is why they might look empty if you check at the wrong time.

**A limitation we found:** Two of the specific bet types we'd most like to use —
number of catches, and number of rush attempts, in a single game — exist as
categories on Kalshi but we have never yet seen an actual live or settled bet in
either category. It's possible these simply haven't launched for this season yet.
We'll know for sure once the regular season starts.

### Betting lines — DraftKings (via The Odds API)

**What it is:** A tool that reads betting lines directly from DraftKings
Sportsbook, since DraftKings itself doesn't offer a public way to check its own
prices programmatically.

**What we get from it:** DraftKings' prices on player stats like receptions, rush
attempts, and pass attempts — the same categories our research found to be most
predictable (see Section 2).

**Why we trust it, with a caveat:** This service is well-established and widely
used, but it's not free at high volume — we get 500 free lookups per month, and
checking odds for a full week of games can use up a meaningful chunk of that. We
built a safety check into our system that calculates the cost *before* running a
search and refuses to run if it would use too much of our monthly allowance, so we
never accidentally run out partway through the season.

**Why we use it at all, alongside Kalshi:** We place our actual bets on
DraftKings, since that's the sportsbook available to us. Kalshi doesn't require
this — checking its prices is free and unlimited — but Kalshi doesn't (yet) offer
some of the specific weekly bets we want. This tool exists so we can see exactly
what price DraftKings offered right before a game started, which lets us check our
own performance later (see "Closing Line Value" below).

### Why we record prices *twice per game day*, and what "Closing Line Value" means

This is one of the most important ideas in sports betting, and it's worth
understanding even if you're not the one placing bets.

A betting line moves between when it's first posted and when the game starts,
because other people are betting on it and the price adjusts. The final price,
right before the game starts, is called the **closing line** — and it's generally
accepted as the single best available estimate of the true odds, because it's had
the most time to absorb everyone's information.

Here's the trick: if you consistently get a *better* price than where the line
eventually closes, you're consistently right about something the market hadn't
figured out yet — **even before you know whether any individual bet won or
lost.** This is called having positive **Closing Line Value**, or CLV.

Why does this matter more than just tracking wins and losses? Because win/loss
records are noisy — you can make 20 great bets and lose most of them just from bad
luck, or make 20 bad bets and get lucky. It takes hundreds of bets before win/loss
records become a trustworthy signal. CLV, on the other hand, starts becoming
meaningful after only a few dozen bets, because it's measuring something more
direct: were you right about the price before the market corrected itself?

This is why Section 1 has you recording prices around *each* game day, not just
once a week — you want a number from early in the week and another right before
that specific game's kickoff, for every day that has games. Without both numbers,
we'd have no way to measure this at all, and betting lines can't be recovered
after the fact once a game has started — so missing this data on any given game
day is a mistake that day can't undo later, even if you catch every other game
day that week correctly.
