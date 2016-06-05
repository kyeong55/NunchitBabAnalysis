USER=$1

for depth in $*
do
if [ $depth != $1 ]
then
python NunchitBabAnalysis.py sample/$USER-bab-sample-60000.csv $depth
python NunchitBabAnalysis.py sample/$USER-bab-sample-180000.csv $depth
python NunchitBabAnalysis.py sample/$USER-bab-sample-300000.csv $depth
fi
done

#python NunchitBabAnalysis.py sample/$USER-bab-sample-60000.csv $depth2
#python NunchitBabAnalysis.py sample/$USER-bab-sample-180000.csv $depth2
#python NunchitBabAnalysis.py sample/$USER-bab-sample-300000.csv $depth2

#python NunchitBabAnalysis.py sample/$USER-bab-sample-60000.csv $depth3
#python NunchitBabAnalysis.py sample/$USER-bab-sample-180000.csv $depth3
#python NunchitBabAnalysis.py sample/$USER-bab-sample-300000.csv $depth3
