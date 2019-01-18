package require pbctools 
pbc wrap -centersel protein -center com -compound res -all
proc fitframes { molid seltext } {
  set ref [atomselect $molid $seltext frame 0]
  set sel [atomselect $molid $seltext]
  set all [atomselect $molid all]
  set n [molinfo $molid get numframes]
   
  for { set i 1 } { $i < $n } { incr i } {
    $sel frame $i
    $all frame $i
    $all move [measure fit $sel $ref]
  }
  return
}
fitframes top backbone
animate write pdb mds_prep/$argv.pdb beg 0 end 0
animate write dcd mds_prep/$argv.dcd beg 1 end -1
quit

