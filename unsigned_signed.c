#include <stdio.h>

int main()
{
	int m = 1;
	unsigned short a = 1, c = 0;
	printf("the size of character is %lu \n",sizeof(char));
/*	for(m;m - sizeof(char) >= 0;m--)
		printf("testing\n");
*/
	unsigned x = 0;
	signed y = -1;
	printf("this is the results %d \n",(y>x));
	int b = -1;
	c = (unsigned short) b;

	printf("%d \n",c);
	printf("%u \n",c);
	return 0;
}


