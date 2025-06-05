% Load the .mat file
filePath = 'D:\Advanced Transmission\齿轮低频振动项目\程序\源数据\20201028扭矩1-454-平板-1-2000rpm\20201028扭矩1-454-平板-1-2000rpm\Throughput\3027_217_Data_Chan16.mat';
data = load(filePath);

% Display detailed information about the variables in the .mat file
disp('Detailed information about variables in the .mat file:');
whos('-file', filePath);

x=1:1:282624;


% Plot the signal to visualize
figure;
plot(x,Data1);
title('Raw Signal');
xlabel('Sample Index');
ylabel('Amplitude');
grid on;

