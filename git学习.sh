git学习
git config --global user.name "He Jinyang"
git config --global user.email hejinyang14@gmail.com

git init push
touch README

#查看仓库当前的状态
git status
#简短
git status -s(git status --short)

#工作区到暂存区
git add README 
#暂存区到分支
git commit -m "XXX" 

#工作区(work dict)和暂存区(stage)的比较
git diff

#暂存区(stage)和分支(master)的比较
git diff --cached

#历史记录
git log
#简化版本
git log --pretty=oneline

#HEAD表示当前版本,上一个版本就是HEAD^
#回退版本
git reset --hard HEAD^
#亦是回退版本
#可用于回退之后反悔
git reset --hard 3628164


#查找回退版本的id
git reflog


#当你改乱了工作区某个文件的内容，想直接丢弃工作区的修改时，用命令
git checkout -- file

#当你不但改乱了工作区某个文件的内容，还添加到了暂存区时，想丢弃修改，分两步，
#第一步用命令
git reset HEAD file
#就回到了场景1，第二步按场景1操作
git checkout -- file

#本地删除文件
rm test.txt
#从版本库中删除该文件
git rm test.txt
git commit -m "remove test.txt"

#恢复文件
git checkout -- test.txt