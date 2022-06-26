% Divide the raw simulated EEG data into left and right conditions 

mydata_p = mydata(:, 2:end); 
leftdata = mydata(:,find(mydata_p(33,:) == 2)); 
rightdata = mydata(:,find(mydata_p(33,:) == 1));

leftdata = leftdata(1:32, :); 
rightdata = rightdata(1:32, :); 

save('leftdata.mat', 'leftdata'); 
save('rightdata.mat', 'rightdata'); 