bash <<'BASH'
set -eo pipefail
mkdir -p /root/workspace
nohup bash -c '
set -eo pipefail
WORKDIR=/mnt/afs-ptdata-shared/sxr/dltrain-mcore018
JOB=interns2-redpajama-adamw-dist-20261126T134623Z-380481
RUN_ID="$JOB-steps500"
LOG="$WORKDIR/logs/$JOB-launcher.log"
test -f "$LOG"
printf "Waiting for the final result bundle: %s\n" "$RUN_ID"
until grep -q "^ADAMW_PIPELINE_EXIT_CODE=" "$LOG"; do
    sleep 10
done
SOURCE="$WORKDIR/runs/$RUN_ID.zip"
DESTINATION="/root/workspace/$RUN_ID.zip"
cp "$SOURCE" "$DESTINATION.part"
cmp "$SOURCE" "$DESTINATION.part"
mv "$DESTINATION.part" "$DESTINATION"
grep "^ADAMW_PIPELINE_EXIT_CODE=" "$LOG"
sha256sum "$DESTINATION"
printf "DOWNLOAD_READY: %s\n" "$DESTINATION"
' > /root/workspace/interns2-result-copy.log 2>&1 < /dev/null &
printf 'COPY_WAITER_PID=%s\n' "$!"
printf '%s\n' 'Watch: tail -F /root/workspace/interns2-result-copy.log'
BASH
