#!/bin/bash -e

exec ffmpeg -f lavfi -i "
    color@generate_white=color=white,
    negate@negate_odd_frames=enable='mod(n, 2)'
" "$@"
