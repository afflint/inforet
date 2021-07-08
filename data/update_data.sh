#!/bin/bash

for url in $(cat ./data_links.txt); do
    curl -sSLO "$url"
done
