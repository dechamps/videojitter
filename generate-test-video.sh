#!/bin/bash -e

exec ffmpeg -f lavfi -i "
    color@generate_white=color=white,
    negate@negate_odd_frames=enable='mod(n, 2)',
    setpts@duplicate_random_frames='if(isnan(PREV_OUTPTS), PTS, PREV_OUTPTS + if(gt(random(0), 0.5), 1, 2) / (FRAME_RATE * TB))'
" -fps_mode cfr "$@"
