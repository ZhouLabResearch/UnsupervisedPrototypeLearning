function stop = proto_output(info)

stop = false;

persistent cnt accs losss d1
global acc loss handles

if info.State == "start"
    d1 = datetime;
    cnt = 0;
    accs = [];
    losss = [];

elseif ~isempty(info.TrainingLoss)

    if handles.run == 0
        stop = true;
        return
    end
    % plot acc and loss
    accs = [ accs acc ];
    losss = [ losss loss ];
    cnt = cnt + 1;
    figure(1)
    subplot(2,3,1)
    plot(1:cnt,accs)
    d2 = datetime - d1;
    title(['timer: ',string(d2),...
        'acc:', num2str(acc)])
    grid on
    subplot(2,3,4)
    plot(1:cnt,losss)
    title(['loss: ',num2str(loss)]);
    grid on

%     if cnt == 0
%         stop = true;
%     end

%     M = getframe(gcf);
%     im = frame2im(M);
%     [imind,cm] = rgb2ind(im,256);
%     if cnt == 3
%         imwrite(imind,cm,'protocircles.gif','gif','DelayTime',0.2, 'Loopcount',inf);
%     elseif mod(cnt,3) == 0
%         imwrite(imind,cm,'protocircles.gif','gif','DelayTime',0.2,'WriteMode','append');
%     end

%     if handles.run == 0
%         stop = true;
%     end

%     if ~isempty(info.) && cnt == 5
%         disp(YValidation(1:5))
%     end
    
end

% function stop_cb(hObj,~)
%     handles.run=0;
%     guidata(hObj,handles)
% end

end