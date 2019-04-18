#!/bin/bash

###############################################################################
#  This bash script runs MD simulations as replica in a serial fashion with   #
# desmond and automatically converts the output with vmd by centering the     #
# protein in the water box and by aligning on heavy atoms of the protein of   #
# the first frame. Place this script in the folder with the cms-, msj- and    #
# cfg-file and change variables according to your system setup and project.   #
# Released under the GNU Public Licence v2.                                   #
###############################################################################

# @author david schaller

# customizable variables
SCHRODINGER=/software/desmond/2018-3 # directory of schrodinger installation
VMD=/software/vmd/vmd-1.9.3/vmd # path to vmd executable
name=name # name of simulation
repeats=10 # number of replica
counter=0 # first replica, can be changed to 5 if, e.g 5 mds have been simulated already and one wants to run more
cms=desmond_md_job_1.cms # name of cms file
msj=desmond_md_job_1.msj # name of msj file
cfg=desmond_md_job_1.cfg # name of cfg file

# create tcl-file for vmd conversion
echo "Creating temporary vmd conversion file ..."
if [ -f md_conversion.tcl ]; then
  rm md_conversion.tcl
fi
cat > md_conversion.tcl << EOF
package require pbctools
pbc wrap -centersel protein -center com -compound res -all
proc fitframes { molid seltext } {
  set ref [atomselect \$molid \$seltext frame 0]
  set sel [atomselect \$molid \$seltext]
  set all [atomselect \$molid all]
  set n [molinfo \$molid get numframes]

  for { set i 1 } { \$i < \$n } { incr i } {
    \$sel frame \$i
    \$all frame \$i
    \$all move [measure fit \$sel \$ref]
  }
  return
}
fitframes top backbone
animate write pdb mds_prep/\$argv.pdb beg 0 end 0
animate write dcd mds_prep/\$argv.dcd beg 1 end -1
quit
EOF

# start simulations
if [ ! -d mds_prep ]; then
  mkdir mds_prep
fi
while ((${counter} < ${repeats}))
do
  if [ -d ${counter} ]; then
    rm -r ${counter}
  fi
  mkdir ${counter}
  echo "Copying files for ${counter} ..."
  cp ${msj} ${cms} ${counter}
  sed "s/seed = 2007/seed = $((2007+${counter}))/" ${cfg} > ${counter}/${cfg}
  cd ${counter}
  echo "Running replica ${counter} ..."
  ${SCHRODINGER}/utilities/multisim \
    -JOBNAME ${name}_${counter} \
    ${cms} \
    -m ${msj} \
    -c ${cfg} \
    -o ${name}_${counter}-out.cms \
    -mode umbrella \
    -set 'stage[1].set_family.md.jlaunch_opt=["-gpu"]' \
    -WAIT
  cd ..
  echo "Converting replica ${counter} ..."
  ${VMD} \
    -f ${counter}/${name}_${counter}-in.cms ${counter}/${name}_${counter}_trj/clickme.dtr \
    -dispdev text \
    -e md_conversion.tcl \
    -eofexit \
    -args ${counter}
  counter=$(($counter+1))
done

# remove temporary vmd conversion script
rm md_conversion.tcl
