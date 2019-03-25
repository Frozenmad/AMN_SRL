#include <stdio.h>
#include <stdlib.h>
#define min(x,y,z) (x>y?(y>z?z:y):(x>z?z:x));

void get_distance_between_lists(int* s1, int* s2, int*a1, int*a2, int m, int n, int* dists) {
	int i;
	int idx1, idx2, a, b, j;
	idx1 = 0;
	idx2 = 0;
	for (a=0; a<m; a++){
		idx2 = 0;
		for (b=0; b<n; b++){
			int distances[a1[a]+1][a2[b]+1];
			for (i = 0; i <= a1[a]; i++)
				distances[i][0] = i;
			for (j = 0; j <= a2[b]; j++)
				distances[0][j] = j;
			for (i = 1; i <= a1[a]; i++)
				for (j = 1; j <= a2[b]; j++)
					if (s1[idx1+i-1] == s2[idx2+j-1])
						distances[i][j] = distances[i-1][j-1];
					else
						distances[i][j] = 1 + min(distances[i-1][j], distances[i][j-1], distances[i-1][j-1]);
			dists[a*n+b] = distances[a1[a]][a2[b]];
			
			idx2 += a2[b];
		}
		idx1 += a1[a];
	}
}


int main(){

	int x1[4] = {20,20,20,20};
	int x2[9] = {30,30,30,30,30,30,30,30,30};
	int y1[2] = {2,2};
	int y2[3] = {3,3,3};
	int b[6] = {0,0,0,0,0,0};
	get_distance_between_lists(x1,x2,y1,y2,2,3,b);

	int i;
	for(i=0;i<2*3;i++)
		printf("%d\n", b[i]);


return 0;
}

