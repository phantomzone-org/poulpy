#!/bin/sh

# this script generates one tag if there is a version change in manifest.yaml
cd `dirname $0`/..
if [ "v$1" = "v-y" ]; then
  echo "production mode!";
fi
changes=`git diff HEAD~1..HEAD -- manifest.yaml | grep 'version:'`
oldversion=$(echo "$changes" | grep '^-version:' | cut '-d ' -f2)
version=$(echo "$changes" | grep '^+version:' | cut '-d ' -f2)
echo "Versions: $oldversion --> $version"
if [ "v$oldversion" = "v$version" ]; then
   echo "Same version - nothing to do"; exit 0;
fi
if [ "v$1" = "v-y" ]; then
  git config user.name github-actions
  git config user.email github-actions@github.com
  git tag -a "v$version" -m "Version $version"
  git push origin "v$version"
else
cat <<EOF
  # the script would do:
  git tag -a "v$version" -m "Version $version"
  git push origin "v$version"
EOF
fi

