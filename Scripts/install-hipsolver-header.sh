if (( $# != 3 ))
then
    echo "usage: $(basename $0) source_dir bindir inst_prefix" >&2
    exit 1
fi

srcdir=$1
bindir=$2
instdir=$3

mkdir -p $bindir/include
mkdir -p $bindir/include/internal
cp $srcdir/library/include/hipsolver.h $bindir/include/
cp $srcdir/library/include/internal/hipsolver-types.h $bindir/include/internal
cp $srcdir/library/include/internal/hipsolver-functions.h $bindir/include/internal
cp $srcdir/library/include/internal/hipsolver-compat.h $bindir/include/internal
cp $srcdir/library/include/internal/hipsolver-refactor.h $bindir/include/internal

mkdir -p $instdir/share/licenses
cp $srcdir/LICENSE.md $instdir/share/licenses/hipsolver.h.LICENSE.md
cp $srcdir/LICENSE.md $instdir/share/licenses/hipsolver-types.h.LICENSE.md