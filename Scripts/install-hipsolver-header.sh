if (( $# != 3 ))
then
    echo "usage: $(basename $0) source_dir bindir inst_prefix" >&2
    exit 1
fi

srcdir=$1
bindir=$2
instdir=$3

cp -r $srcdir/library/include $bindir

mkdir -p $instdir/share/licenses
cp $srcdir/LICENSE.md $instdir/share/licenses/hipsolver.h.LICENSE.md
cp $srcdir/LICENSE.md $instdir/share/licenses/hipsolver-types.h.LICENSE.md

