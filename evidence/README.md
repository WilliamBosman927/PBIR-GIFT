# Compact evidence bundle

This directory intentionally contains only lightweight, inspectable evidence:

- Stage-A sweep summary and frozen parameter selection;
- Stage-B through Stage-H analysis bundles;
- Stage-I final comparison tables and metadata;
- Stage-C/Stage-F frozen shared code artifacts used by the paired replay.

It excludes model checkpoints and the full locked result tree. Consequently, it supports inspection of reported calculations but does not replace the full archived evidence needed to rerun every lock verification. The original local/NAS result tree must be preserved separately.

