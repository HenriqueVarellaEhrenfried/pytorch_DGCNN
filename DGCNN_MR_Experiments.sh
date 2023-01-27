
# ./run_DGCNN.sh MR-GOW-TAG-None 0


_10KGNAD=('1-10kGNAD-Only-Order' \
'2-10kGNAD-Order-Circular' \
'3-10kGNAD-Binary-Tree' \
'4-10kGNAD-Dependency-Tree' \
'5-10kGNAD-Dependency-Tree-Order' \
'6-10kGNAD-Dependency-Tree-Order-Multigraph' \
'7-10kGNAD-Dependency-Tree-Self' \
'8-10kGNAD-Dependency-Tree-Order-Self' \
'9-10kGNAD-Dependency-Tree-Order-Multigraph-Self' \
'10-10kGNAD-RedBlack-Tree' \
'11-10kGNAD-AVL-Tree' \
'12-10kGNAD-GOW'  
)


B2W=('1-B2W-Rating-Only-Order' \
'2-B2W-Rating-Order-Circular' \
'3-B2W-Rating-Binary-Tree' \
'4-B2W-Rating-Dependency-Tree' \
'5-B2W-Rating-Dependency-Tree-Order' \
'6-B2W-Rating-Dependency-Tree-Order-Multigraph' \
'7-B2W-Rating-Dependency-Tree-Self' \
'8-B2W-Rating-Dependency-Tree-Order-Self' \
'9-B2W-Rating-Dependency-Tree-Order-Multigraph-Self' \
'10-B2W-Rating-RedBlack-Tree' \
'11-B2W-Rating-AVL-Tree' \
'12-B2W-Rating-GOW'  
)

MR=('1-MR-Only-Order' \
'2-MR-Order-Circular' \
'3-MR-Binary-Tree' \
'4-MR-Dependency-Tree' \
'5-MR-Dependency-Tree-Order' \
'6-MR-Dependency-Tree-Order-Multigraph' \
'7-MR-Dependency-Tree-Self' \
'8-MR-Dependency-Tree-Order-Self' \
'9-MR-Dependency-Tree-Order-Multigraph-Self' \
'10-MR-RedBlack-Tree' \
'11-MR-AVL-Tree' \
'12-MR-GOW'  
)

OHSUMED=('1-Ohsumed-Only-Order' \
'2-Ohsumed-Order-Circular' \
'3-Ohsumed-Binary-Tree' \
'4-Ohsumed-Dependency-Tree' \
'5-Ohsumed-Dependency-Tree-Order' \
'6-Ohsumed-Dependency-Tree-Order-Multigraph' \
'7-Ohsumed-Dependency-Tree-Self' \
'8-Ohsumed-Dependency-Tree-Order-Self' \
'9-Ohsumed-Dependency-Tree-Order-Multigraph-Self' \
'10-Ohsumed-RedBlack-Tree' \
'11-Ohsumed-AVL-Tree' \
'12-Ohsumed-GOW'  
)


for I in ${_10KGNAD[*]}; 
do \
echo "!! >> WORKING WITH [${I}] << !! ";
echo "";
echo -n "./run_DGCNN.sh ${I} 0 ";
echo "";
echo "--------------------------------";
echo "";
echo "";

done ; 
echo "======================================";
echo "";

for I in ${B2W[*]}; 
do \
echo "!! >> WORKING WITH [${I}] << !! ";
echo "";
echo -n "./run_DGCNN.sh ${I} 0 ";
echo "";
echo "--------------------------------";
echo "";

done ; 
echo "======================================";
echo "";

for I in ${MR[*]}; 
do \
echo "!! >> WORKING WITH [${I}] << !! ";
echo "";
echo -n "./run_DGCNN.sh ${I} 0 ";
echo "";
echo "--------------------------------";
echo "";

done ; 
echo "======================================";
echo "";

for I in ${OHSUMED[*]}; 
do \
echo "!! >> WORKING WITH [${I}] << !! ";
echo "";
echo -n "./run_DGCNN.sh ${I} 0 ";
echo "";
echo "--------------------------------";
echo "";

done ; 
echo "======================================";
echo "";
echo