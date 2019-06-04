# Git workflow

After a modif on file.cu:

	git add file.cu
	git commit -m "Explicit message about the modif"
	git push origin master

To pull modifications from origin master

	git pull --rebase origin master

At the end of the day

	git status

to see the modified files not staged for commit

	git add modified_file_to_keep.cu
	git rm file_to_remove.cu
	git commit -m "Explicit message about the work done"

If everything works and the feature has to be merged

	git push origin master

