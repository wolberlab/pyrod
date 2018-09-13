#!/bin/bash

repeats=10
name=name

counter=0
mkdir mds_prep
while (("$counter" < "$repeats"))
do
if [ -d "$counter" ]
then
rm -r $counter
fi
mkdir $counter
cp desmond_md_job_1.msj desmond_md_job_1.cms $counter
sed "s/seed = 2007/seed = $((2007+$counter))/" desmond_md_job_1.cfg > $counter/desmond_md_job_1.cfg
cd $counter
$SCHRODINGER/utilities/multisim -JOBNAME ${name}_${counter} -m desmond_md_job_1.msj -c desmond_md_job_1.cfg -o ${name}_${counter}-out.cms desmond_md_job_1.cms -mode umbrella -set 'stage[1].set_family.md.jlaunch_opt=["-gpu"]' -WAIT
vmd -f ${name}_${counter}-in.cms ${name}_${counter}_trj/clickme.dtr -dispdev text -e ../md_prep.tcl -eofexit -args $counter
cd ..
counter=$(($counter+1))
done
