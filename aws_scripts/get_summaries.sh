echo "getting tensorboard summaries from an AWS EC2 instance"

echo "using the key:"
echo "$1"

echo "to access the instance:"
echo "$2"

DATE=`date +%Y-%m-%d`
TIME=`date +%H%M`

echo $DATE
echo $TIME

echo scp -i ~/.ssh/"$1" -r ubuntu@"$2":/home/ubuntu/deep-pong/data_summaries ../aws_runs/"$DATE"/"$TIME"


DIR0="../aws_runs/"$DATE""
echo $DIR0

if [ ! -d "$DIR0" ]; then
  mkdir $DIR0
fi

DIR1=""$DIR0"/"$TIME""
echo $DIR1

if [ ! -d "$DIR1" ]; then
  mkdir $DIR1
fi

scp -i ~/.ssh/"$1" -r ubuntu@"$2":/home/ubuntu/deep-pong/data_summaries "$DIR1"
