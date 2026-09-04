# Data

The organizer corpus is intentionally not committed. Obtain the released
ViFinQA data through the competition distribution and place it here:

```text
data/raw/vifinqa/
├── code_stock.csv
├── questions/questions.jsonl
└── financial_statements/<TICKER>/<YEAR>/<REPORT_ID>/
    └── <REPORT_ID>_extracted.txt
```

The pipeline derives `data/derived/line_items.json` and local annotations during
experiments; those files are ignored. The small `benchmark_v2` development
annotation set is the one intentional exception and is published for teammate
review. No raw financial reports are needed to inspect the architecture or run
unit tests, but they are required to rebuild a full candidate pool or submission
from scratch.
