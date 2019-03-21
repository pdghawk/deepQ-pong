echo "uploading code to an AWS EC2 instance"

echo "one should make the directory deep-pong/code available on the instance before running"

echo "using the key:"
echo "$1"

echo "to access the instance:"
echo "$2"

scp -i ~/.ssh/"$1" -r ../code/DQN ubuntu@"$2":/home/ubuntu/deep-pong/code/DQN
scp -i ~/.ssh/"$1" ../code/simple_loops.py ubuntu@"$2":/home/ubuntu/deep-pong/code
scp -i ~/.ssh/"$1" ../code/play_from_ckpt.py ubuntu@"$2":/home/ubuntu/deep-pong/code
scp -i ~/.ssh/"$1" ../code/single_train.py ubuntu@"$2":/home/ubuntu/deep-pong/code
