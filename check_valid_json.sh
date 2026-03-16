#!/usr/bin/env bash

JSON_DIR=$1

# Loop over all *.json files (recursively if needed)
find "$JSON_DIR" -type f -name '*.json' | while read -r file; do
  if ! jq empty "$file" >/dev/null 2>&1; then
    echo "Invalid JSON: $file"
  fi
done
