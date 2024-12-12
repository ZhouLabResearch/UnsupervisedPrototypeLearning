classdef customRegressionLayerKm < nnet.layer.RegressionLayer ...
        & nnet.layer.Acceleratable
    % Example custom regression layer with mean-absolute-error loss.

    methods
        function layer = customRegressionLayerCCEacc(name)
            % layer = maeRegressionLayer(name) creates a
            % mean-absolute-error regression layer and specifies the layer
            % name.

            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = 'Mean absolute error';
        end

        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the MAE loss between
            % the predictions Y and the training targets T.
            global xproto yproto
            y = permute(Y,[ 4 3 2 1 ]); % logits
            t = permute(T,[ 4 3 2 1 ]); % labels
            t = t(:,1);
            R = size(y,1);

            % x and y coordinates of output
            xm = y(:,1);
            ym = y(:,2);

            % calculate each prototype (mean per class)
            % kmeans here instead
            xproto = [];
            yproto = [];

            try
                X = [xm, ym];
                X = extractdata(X);

                %%%%%%%%%%%%%%%%%
                % kmeans
                % X is 1024 x 2 gpuArray, output space
                NN=2;
                opts = statset('Display','off');
                % C is centroids of each cluster, the mean m x m where m is
                % number of clusters, x and y; x and y of prototype gpuArray
                % t is n x 1 double index of which element belongs to which cluster
                [t,C] = kmeans(X,NN,'Distance','sqeuclidean',...
                    'Replicates',10,'Options',opts);
                %%%%%%%%%%%%%%%%%

                xproto = C(:,1);
                yproto = C(:,2);

            catch
                loss = Y(1,1,1,1);
                disp('error')
                return
            end


            % get the distance between all prototypes
            one = 1; two = 2;
            maxdist = 0;
            maxsum = 0;
            for k = 1:length(xproto)
                %                 backx = xvec; backy = yvec;
                currx = xproto(k); curry = yproto(k);
                %                 xvec(k) = []; yvec(k) = [];

                distance = sqrt( (currx-xproto).^2 +...
                    (curry-yproto).^2 );

                % get max distance
                [ v p ] = max(distance);
                if v > maxdist
                    one = k; two = p;
                    maxdist = v;
                end

                distance = sum(distance);
                % get all mean distances
                % maxsum = maxsum + distance/(R-1); % /(R-1)
                %                 maxsum = maxsum + 1/(distance/(length(xproto)-1) +0.001);
                maxsum = maxsum + distance;
                %                 xvec = backx; yvec = backy;
            end

            % get the distance from each output to each prototype
            % from each xm ym pair, to each xproto yproto pair
            protodist = [];
            %             msum = 0;
            for k = 1:length(xproto)
                distance = sqrt( (xm - xproto(k)).^2 +...
                    (ym - yproto(k)).^2 );
                %                 if k == 1
                %                 mdist = sqrt( (xm - mproto(1)).^2 +...
                %                                  (ym - mproto(2)).^2 );
                %                 msum = msum + sum(mdist);
                %                 end
                protodist = [ protodist; distance' ];
            end
            %             msum = msum/size(protodist,2);
            protoback = protodist;
            % normalize the rows of protodist
            gam = 0.01;
            protodist = exp(-gam.*protodist);
            protonorm = protodist./sum(protodist);

            % get the row of each columns class
            tproto = onehotencode(categorical(t),2);
            tproto = logical(tproto);

            % get the prototype loss
            % get the distance to each outputs class' prototype
            global loss acc
            cls = 0;
            clv = 0;
            t = t';
            minsum = 0;
            for k = 1:length(t)
                curr = t(k);
                minsum = minsum + protodist(curr,k);

                %%% acc part can remove
                [ v p ] = min(protoback(:,k));
                cls = cls + ( p == curr ); % correct predictions
                %                 acc = cls/length(t);
                [ v p ] = max(protonorm(:,k));
                clv = clv + and(( p == curr ),...
                    (v > 0.90)); % correct predictions
                %%% acc part can remove
            end
            acc = cls/length(t);
            acv = clv/length(t);
            %             acv = (size(protonorm,2) - clv)/size(protonorm,2);

            %             gfun = @(x) gather(double(extractdata(x)));
            % get the sum of the correct normalized distance
            protonorm = protonorm';
            loss = sum(protonorm(tproto));
            lam = 0.01;
            % min = -log(min)
            % max = log(max) ?
            %             loss = -abs(-log(sqrt(loss))); %-...
            %lam*minsum; %  - gfun(log(maxsum))
            idx = t==1;
            ss = y(idx,:); s1 = size(ss,1);
            idx = t==2;
            ss = y(idx,:); s2 = size(ss,1);
            ms = max(s1,s2);
            % loss = -log(loss);% + lam*minsum;
            %             loss = -maxsum -lam*minsum + Y(1,1,1,1)*0;
            loss = -log(loss) -maxsum;

            figure(1)
            subplot(2,3,[ 2 3 5 6 ])
            for z = 1:NN
                idx = t==z;
                plot(y(idx,1),y(idx,2),'o')
                hold on
            end
            %             plot(xmc,ymc,'b*')
            plot(xproto,yproto,'k*','linewidth',5)
            %             title(['A: ',num2str(loss),' B: ',...
            %                 num2str(circsum),' C: ',num2str(- lam*minsum)])
            title(['acc rr > 0.90: ',num2str(acv)]);
            % syms x y
            %             ff = (x - gfun(xmc))^2 + (y - gfun(ymc))^2 - gfun(dist)^2;
            %             fimplicit(ff,'Color','k')
            grid on
            % g = gcf;
            % gcf.Color = [ 1 1 1 ];

            pause(0.01)
            hold off

        end
    end
end
