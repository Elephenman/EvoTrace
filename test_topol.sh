#!/bin/bash
# test_topol.sh — 验证 tleap 建拓扑 + 受体/配体剥离（区间法）+ make_restraints mask 编号
source ~/md/activate_md.sh >/dev/null 2>&1
PARMED=$HOME/miniconda3/envs/md/bin/parmed
BASE=~/md; SYS=WT__S1; TD=$BASE/topol_test/$SYS; mkdir -p "$TD"
IN=$BASE/inputs/six/v${SYS}.pdb
cat > "$TD/leap.in" <<EOF
source leaprc.protein.ff14SB
source leaprc.DNA.OL15
source leaprc.water.tip3p
loadoff atomic_ions.lib
loadamberparams $AMBERHOME/dat/leap/parm/frcmod.ions1lm_1264_tip3p
loadamberparams $BASE/ff/mn_cm12-6.frcmod
mol = loadpdb $IN
check mol
desc mol
solvateoct mol TIP3PBOX 12.0
addions mol Na+ 0
addions mol Cl- 0
saveamberparm mol $TD/v${SYS}.parm7 $TD/v${SYS}.rst7
quit
EOF
tleap -f "$TD/leap.in" > "$TD/tleap.log" 2>&1 && echo TLEAP_OK || { echo TLEAP_FAIL; tail -50 "$TD/tleap.log"; exit 1; }
python3 - <<PY
import parmed as pmd
t=pmd.load_file("$TD/v${SYS}.parm7")
mn=[a for a in t.atoms if a.name=="MN"]
print(f"FULL n_atom={len(t.atoms)} totQ={sum(a.charge for a in t.atoms):.2f} MnQ={mn[0].charge if mn else 'MISSING'} MnMass={mn[0].mass if mn else 'NA'}")
res=[(r.idx+1,r.name) for r in t.residues]
print("FULL nres=",len(res),"head=",res[:2],"tail=",res[-4:])
PY
# make_restraints -> masks.env (含 PROT_RANGE / DNA_RANGE / MN_RES)
python3 $BASE/md/make_restraints.py "$TD/v${SYS}.parm7" "$TD/v${SYS}.rst7" "$TD"
source "$TD/masks.env"
echo "PROT_RANGE=$PROT_RANGE DNA_RANGE=$DNA_RANGE MN_RES=$MN_RES"
# 气相复合体 cp（去水/离子）
cat > "$TD/cp.in" <<EOF
parm $TD/v${SYS}.parm7
strip :WAT,:Na+,:Cl-
outparm $TD/cp.parm7
quit
EOF
# 受体 = 蛋白 + Mn
cat > "$TD/rec.in" <<EOF
parm $TD/cp.parm7
strip :${DNA_RANGE}
outparm $TD/rec.parm7
quit
EOF
# 配体 = DNA
cat > "$TD/lig.in" <<EOF
parm $TD/cp.parm7
strip :${PROT_RANGE},:${MN_RES}
outparm $TD/lig.parm7
quit
EOF
$PARMED -i "$TD/cp.in"  >/dev/null 2>&1 && echo CP_OK
$PARMED -i "$TD/rec.in" >/dev/null 2>&1 && echo REC_OK
$PARMED -i "$TD/lig.in" >/dev/null 2>&1 && echo LIG_OK
python3 - <<PY
import parmed as pmd
DNA={"DA","DC","DG","DT","DA5","DT5","DA3","DT3","DC5","DG5","DC3","DG3","DU","DU5","DU3"}
for tag,f in [("REC","$TD/rec.parm7"),("LIG","$TD/lig.parm7"),("CP","$TD/cp.parm7")]:
    t=pmd.load_file(f)
    names=[r.name for r in t.residues]
    ndna=sum(1 for n in names if n in DNA)
    nwat=names.count("WAT"); nprot=sum(1 for n in names if n not in DNA and n not in ("WAT","Na+","Cl-","MN"))
    nmn=names.count("MN")
    print(f"{tag:5s} nres={len(names):4d} n_atom={len(t.atoms):6d} PROT={nprot} DNA={ndna} MN={nmn} WAT={nwat}")
PY
