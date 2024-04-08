export DATA_ROOT=/projects/katefgroup/datasets/scannet_sens/groundtruth # path to scannet sens data (change here)
export TARGET=/projects/katefgroup/language_grounding/SEMSEG_100k/frames_square_highres   # data destination (change here)

reader() {
    filename=$1
    frame_skip=20

    scene=$(basename -- "$filename")
    scene="${scene%.*}" 
    echo "Find sens data: $filename $scene"
    python -u reader.py --filename $filename --output_path $TARGET/$scene --frame_skip $frame_skip 
}

export -f reader

parallel -j 16 --linebuffer time reader ::: `find $DATA_ROOT/scene*/*.sens`
# reader '/scratch/scans/scene0191_00/scene0191_00.sens'